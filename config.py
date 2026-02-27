import os
import torch
from pathlib import Path
import json
import argparse
from enum import Enum
from utils.dataset import DatasetValidator
import torch.distributed as tdist
from datetime import datetime

class Config:
    
    class RunMode(Enum):
        INIT = "init"
        PRETRAIN = "pretrain"
        TRAIN = "train"
        INFER = "infer"

    class GeneratorTypes(Enum):
        RESNET = "resnet"
        FASTAI = "fastai"

    class InferMode(Enum):
        TRAINED = "trained"
        INDUSTRIAL = "industrial"

    def __init__(self, configFile: str = ".config.json"):
        self.use_ddp = False
        self.local_rank = 0 #Because of tqdm
        self.pretrained = False
        self.args = self._parse_args()
        self._set_properties_from_arguments()
        try:
            self.device = self._getDevice()
            self.configFile = Path(configFile)
            if self.use_ddp:
                tdist.init_process_group(backend="nccl")

            if self.local_rank == 0:
                self.config = self._precheck_config()

            if self.local_rank != 0:
                self.config = self._load_config()

            if self.mode != Config.RunMode.INIT:
                self.validator = DatasetValidator(self.config["dataset"]["path"])
                self._check_dataset()
                
            self._set_out_filenames()
        except RuntimeError as e:
            print(f"RuntimeError: {e}")
            exit(1)

    def _parse_args(self):
        parser = argparse.ArgumentParser()

        subparsers = parser.add_subparsers(dest="mode", required=True, help="Available commands")

        initParser = subparsers.add_parser("init", help="Initialize dataset for training")

        pretrainParser = subparsers.add_parser("pretrain")
        pretrainParser.add_argument("--imSize", type=int, required=True, choices=[64, 96, 128, 256], help="Image size")
        pretrainParser.add_argument("--batchSize", type=int, required=True, choices=[64, 96, 128, 192, 256, 384], help="Batch size")
        pretrainParser.add_argument("--numEpoch", type=int, required=True, help="Number of epochs to pretrain the model for")
        pretrainParser.add_argument("--generatorType", type=str, required=True, choices=["resnet", "fastai"], help="Generator type to pretrain")

        trainParser = subparsers.add_parser("train")
        trainParser.add_argument("--imSize", type=int, required=True, choices=[64, 96, 128, 256], help="Image size")
        trainParser.add_argument("--batchSize", type=int, required=True, choices=[64, 96, 128, 192, 256, 384], help="Batch size")
        trainParser.add_argument("--numEpoch", type=int, required=True, help="Number of epochs to train the model for")
        trainParser.add_argument("--generatorType", type=str, required=True, choices=["resnet", "fastai"], help="Generator type to use for GAN model")
        trainParser.add_argument("--pretrainGeneratorPath", type=str, required=False, help="Path to the pretrained generator")
        
        inferParser = subparsers.add_parser("infer")
        subInferParser = inferParser.add_subparsers(dest="inferMode", required=True, help="Inferance subcommand")

        trainedInferParser = subInferParser.add_parser("trained")
        trainedInferParser.add_argument("--imSize", type=int, required=True, choices=[64, 96, 128, 256], help="Image size")
        trainedInferParser.add_argument("--modelPath", type=str, required=True, help="Path of the model to infer")
        trainedInferParser.add_argument("--generatorType", type=str, required=True, choices=["resnet", "fastai"], help="Generator type to use for inferance")

        industrialInferParser = subInferParser.add_parser("industrial")
        industrialInferParser.add_argument("--imSize", type=int, required=True, choices=[64, 96, 128, 256], help="Image size")

        args = parser.parse_args()

        return args
    
    def _set_properties_from_arguments(self):
        self.mode = self.RunMode(self.args.mode)
        if hasattr(self.args, "inferMode"):
            self.inferMode = self.InferMode(self.args.inferMode)
        if hasattr(self.args, "generatorType"):
            self.generatorType = self.GeneratorTypes(self.args.generatorType)
        if hasattr(self.args, "imSize"):
            self.imSize = self.args.imSize
        if hasattr(self.args, "batchSize"):
            self.batchSize = self.args.batchSize
        if hasattr(self.args, "numEpoch"):
            self.numEpoch = self.args.numEpoch
    
    def _set_out_filenames(self):
        formattedTime = datetime.now().strftime("%d%m%Y%H%M")
        resultsDir = Path(self.config["output"]["resultsDir"])

        if self.mode == self.RunMode.TRAIN:
            outDir = Path(self.config["output"]["modelDir"])
            os.makedirs(outDir, exist_ok=True)
            modelPath = f"{self.generatorType.value}_epoch{self.numEpoch}_{formattedTime}.pth"
            if hasattr(self.args, "pretrainGeneratorPath"):
                self.pretrained = True
                self.pretrainGeneratorPath = Path(self.args.pretrainGeneratorPath)
                modelPath = "pretrained_" + modelPath
            self.modelDir = os.path.join(outDir, f"model_{formattedTime}")
            # os.makedirs(self.modelDir, exist_ok=True)
            self.modelPath = os.path.join(self.modelDir, modelPath)
        elif self.mode == self.RunMode.PRETRAIN:
            outDir = Path(self.config["output"]["pretrainedModelDir"])
            os.makedirs(outDir, exist_ok=True)
            modelPath = f"{self.generatorType.value}_epoch{self.numEpoch}_{formattedTime}.pth"
            modelPath = "generator_pretrained_" + modelPath
            self.modelDir = os.path.join(outDir, f"model_{formattedTime}")
            # os.makedirs(self.modelDir, exist_ok=True)
            self.modelPath = os.path.join(self.modelDir, modelPath)
        elif self.mode == self.RunMode.INFER:
            if self.inferMode == self.InferMode.TRAINED:
                self.modelPath = Path(self.args.modelPath)
            infareDir = f"run_infer_{formattedTime}"
            resultsDir = os.path.join(resultsDir, infareDir)

        self.resultsDir = resultsDir
        self.runDir = os.path.join(resultsDir, f"run_{self.mode.value}_{formattedTime}")

    def _getDevice(self):
        if 'LOCAL_RANK' in os.environ:
            if self.mode in [self.RunMode.INIT]:
                raise RuntimeError("[CONFIG] Distributed mode cannot be used for initialization!")
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
            self.use_ddp = True
            print(f"[CONFIG] Using multiGPU setup! cuda:{self.local_rank}")
        elif torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"[CONFIG] Using single GPU setup! cuda:0")
        else:
            device = torch.device('cpu')
            print("[CONFIG] Warning: Using CPU!")
            if self.mode in [self.RunMode.TRAIN]:
                raise RuntimeError("[CONFIG] The CPU cannot be used for training of this model!")

        return device
    
    def _check_dataset(self):
        if not self.validator.checkExists():
            raise RuntimeError("[CONFIG] Missing dataset. Initialization required. Run: python3 main.py init")

        datasetInfo = self.validator.getDatasetInfo()
        if not datasetInfo:
            raise RuntimeError("[CONFIG] Missing dataset. Initialization required. Run: python3 main.py init")

        self.config["dataset"]["initialized"] = True
        self.config["dataset"]["trainSize"] = datasetInfo["trainSize"]
        self.config["dataset"]["valSize"] = datasetInfo["valSize"]
        self.config["dataset"]["lastVerified"] = datasetInfo["lastVerified"]
        if self.local_rank == 0:
            self._save_config()

        return True
    
    def _precheck_config(self):
        if self.local_rank == 0:
            default_config = self._get_default_config()

            if self.configFile.exists():
                try:
                    with open(self.configFile, 'r') as f:
                        loaded_config = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    print("[CONFIG] Config file corrupted or empty. Regenerating default config.")

                    with open(self.configFile, 'w') as f:
                        json.dump(default_config, f, indent=4)

                    return default_config
                
                if not isinstance(loaded_config, dict):
                    print("[CONFIG] Config file corrupted. Regenerating default config.")

                    with open(self.configFile, 'w') as f:
                        json.dump(default_config, f, indent=4)

                    return default_config

                updated_config = self._merge_dicts(default_config, loaded_config)

                if updated_config != loaded_config:
                    with open(self.configFile, 'w') as f:
                        json.dump(updated_config, f, indent=4)

                return updated_config
            else:
                print("[CONFIG] Generating default config file!")
                with open(self.configFile, 'w') as f:
                    json.dump(default_config, f, indent=4)

                return default_config
    
    def _load_config(self):
        if self.configFile.exists():
            try:
                with open(self.configFile, 'r') as f:
                    loaded_config = json.load(f)
            except Exception as e:
                raise RuntimeError("[CONFIG] Config file corrupted. Run initialization again with: python3 main.py init")

            return loaded_config
        else:
            raise RuntimeError("[CONFIG] Config file missing. Run initialization again with: python3 main.py init")
        
    def _merge_dicts(self, default, loaded):
        for key, value in default.items():
            if key not in loaded:
                loaded[key] = value
            elif isinstance(value, dict):
                loaded[key] = self._merge_dicts(value, loaded.get(key, {}))
        return loaded
            
    def _get_default_config(self):
        return {
                    "output": {
                        "modelDir": "./modelCheckpoints",
                        "pretrainedModelDir": "./pretrainedModelCheckpoints",
                        "resultsDir": "./results"
                    },
                    "dataset": {
                        "initialized": False,
                        "trainSize": 0,
                        "valSize": 0,
                        "lastVerified": None,
                        "path": "./data/coco"
                    },
                    "results": {
                        "imgToDisplay": [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]
                    }
                }
        
    def _reset_dataset_state(self):
        self.config["dataset"].update({
                        "initialized": False,
                        "trainSize": 0,
                        "valSize": 0,
                        "lastVerified": None,
                        "path": "./data/coco"
                    })

    def _save_config(self):
        tmp_path = self.configFile.with_suffix(".tmp")

        with open(tmp_path, 'w') as f:
            json.dump(self.config, f, indent=4)

        os.replace(tmp_path, self.configFile)
    
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
    def INFER_MODE(self):
        return self.inferMode
    
    @property
    def GENERATOR_TYPE(self):
        return self.generatorType
    
    @property
    def MODEL_PATH(self):
        return self.modelPath
    
    @property
    def RESULTS_DIR(self):
        return self.resultsDir
    
    @property
    def IMAGE_SIZE(self):
        return self.imSize
    
    @property
    def BATCH_SIZE(self):
        return self.batchSize
    
    @property
    def NUM_EPOCH(self):
        return self.numEpoch
    
    @property
    def PRETRAINED_GENERATOR_PATH(self):
        return self.pretrainGeneratorPath
    
    @property
    def IMG_TO_DISPLAY(self):
        return self.config["results"]["imgToDisplay"]
    
    @property
    def PRETRAINED(self):
        return self.pretrained

    @property
    def RUN_DIR(self):
        return self.runDir

    @property
    def MODEL_DIR(self):
        return self.modelDir

CONFIG = Config()