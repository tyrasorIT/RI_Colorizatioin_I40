import os
import torch
from pathlib import Path
import json
import argparse
from enum import Enum
from utils.dataset import DatasetValidator

class Config:
    
    class RunMode(Enum):
        INIT = "init"
        PRETRAIN = "pretrain"
        TRAIN = "train"
        INFER = "infer"

    class GeneratorTypes(Enum):
        RESNET = "resnet"
        FASTAI = "fastai"
        PRETRAINEDRESNET = "pretrainedResnet"

    def __init__(self, configFile: str = ".config.json"):
        self.use_ddp = False
        self.local_rank = 0 #Because of tqdm
        self.args = self._parse_args()
        self.mode = self.RunMode(self.args.mode)
        self.generatorType = self.GeneratorTypes(self.args.generatorType)
        try:
            self.device = self._getDevice()
            self.configFile = Path(configFile)
            self.config = self._load_config()
            if self.local_rank == 0: #Ugly fix, consider moving from global to main config
                self.validator = DatasetValidator(self.config["dataset"]["path"])
                self._checkDataset()
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            exit(1)

    def _parse_args(self):
        parser = argparse.ArgumentParser()

        subparsers = parser.add_subparsers(dest="mode", required=True, help="Available commands")

        initParser = subparsers.add_parser("init", help="Initialize dataset for training")

        pretrainParser = subparsers.add_parser("pretrain")
        pretrainParser.add_argument("--imSize", type=int, required=True, choices=[64, 96, 128, 256], help="Image size")
        pretrainParser.add_argument("--batchSize", type=int, required=True, choices=[64, 96, 128, 256, 384], help="Batch size")
        pretrainParser.add_argument("--numEpoch", type=int, required=True, help="Number of epochs to pretrain the model for")
        pretrainParser.add_argument("--generatorType", type=str, required=True, choices=["fastai", "pretrainedResnet"], help="Generator type to pretrain")

        trainParser = subparsers.add_parser("train")
        trainParser.add_argument("--imSize", type=int, required=True, choices=[64, 96, 128, 256], help="Image size")
        trainParser.add_argument("--batchSize", type=int, required=True, choices=[64, 96, 128, 256, 384], help="Batch size")
        trainParser.add_argument("--numEpoch", type=int, required=True, help="Number of epochs to train the model for")
        trainParser.add_argument("--generatorType", type=str, required=True, choices=["resnet", "fastai", "pretrainedResnet"], help="Generator type to use for GAN model")
        trainParser.add_argument("--pretrainGeneratorPath", type=str, required=False, help="Path to the pretrained generator")
        
        inferParser = subparsers.add_parser("infer")
        inferParser.add_argument("--imSize", type=int, required=True, choices=[64, 96, 128, 256], help="Image size")
        inferParser.add_argument("--modelPath", type=str, required=True, help="Path of the model to infer")
        inferParser.add_argument("--generatorType", type=str, required=True, choices=["resnet", "fastai", "pretrainedResnet"], help="Generator type to use for GAN model")
        inferParser.add_argument("--pretrainGeneratorPath", type=str, required=False, help="Path to the pretrained generator")

        args = parser.parse_args()

        if args.mode in ["train", "infer"]:
            if args.generatorType in ["fastai", "pretrainedResnet"] and not args.pretrainGeneratorPath:
                parser.error("--pretrainGeneratorPath is required when generatorType is 'fastai' or 'pretrainedResnet'")

        return args

    def _getDevice(self):
        if 'LOCAL_RANK' in os.environ:
            if self.mode in [self.RunMode.INIT]:
                raise RuntimeError("[CONFIG] Distributed mode cannot be used for initialization!")
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
            self.use_ddp = True
            if self.local_rank == 0:
                print("[CONFIG] Using multiGPU setup!")
        elif torch.cuda.is_available():
            device = torch.device('cuda:0')
            print("[CONFIG] Using single GPU setup!")
        else:
            device = torch.device('cpu')
            print("[CONFIG] Warning: Using CPU!")
            if self.mode in [self.RunMode.TRAIN]:
                raise RuntimeError("[CONFIG] The CPU cannot be used for training of this model!")

        return device
    
    def _checkDataset(self):
        if self.local_rank == 0:
            if not self.validator.checkExists():
                self.config = self._get_clear_config()
                self._save_config()
                raise RuntimeError("[CONFIG] Missing dataset. Initialization required. Run: python3 main.py init")

            datasetInfo = self.validator.getDatasetInfo()
            if not datasetInfo:
                self.config = self._get_clear_config()
                self._save_config()
                raise RuntimeError("[CONFIG] Missing dataset. Initialization required. Run: python3 main.py init")

            self.config["dataset"]["initialized"] = True
            self.config["dataset"]["trainSize"] = datasetInfo["trainSize"]
            self.config["dataset"]["valSize"] = datasetInfo["valSize"]
            self.config["dataset"]["lastVerified"] = datasetInfo["lastVerified"]

            self._save_config()
            return True
    
    def _load_config(self):
        if self.local_rank == 0:
            if self.configFile.exists():
                with open(self.configFile, 'r') as f:
                    return json.load(f)
            else:
                return self._get_clear_config()
        
    def _get_clear_config(self):
        return {
                    "dataset": {
                        "initialized": False,
                        "trainSize": 0,
                        "valSize": 0,
                        "lastVerified": None,
                        "path": "./data/coco"
                    }
                }

    def _save_config(self):
        if self.local_rank == 0:
            with open(self.configFile, 'w') as f:
                json.dump(self.config, f, indent=4)
    
    @property
    def DEVICE(self):
        return self.device
    
    @property
    def LOCAL_RANK(self):
        return self.local_rank
    
    @property
    def USE_DDP(self):
        return self.use_ddp
    
    @property
    def MODE(self):
        return self.mode
    
    @property
    def GENERATORTYPE(self):
        return self.generatorType

CONFIG = Config()