import numpy as np
import os
import random
import torch
import yaml
from logger import LogLevel
from yacs.config import CfgNode as CN


_C = CN()

# Base config files
_C.BASE = ['']


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = None

""" seq2seq """
_C.MODEL.SEQ2SEQ = CN()
_C.MODEL.SEQ2SEQ.DIM = 512
_C.MODEL.SEQ2SEQ.N_HEADS = 8
_C.MODEL.SEQ2SEQ.N_ENCODER_LAYERS = 6
_C.MODEL.SEQ2SEQ.N_DECODER_LAYERS = 6
_C.MODEL.SEQ2SEQ.DIM_FEEDFORWARD = 2048
_C.MODEL.SEQ2SEQ.DROPOUT = 0.0
_C.MODEL.SEQ2SEQ.SRC_VOCAB_SIZE = None
_C.MODEL.SEQ2SEQ.TGT_VOCAB_SIZE = None
_C.MODEL.SEQ2SEQ.SRC_SEQ_LEN = None
_C.MODEL.SEQ2SEQ.TGT_SEQ_LEN = None
_C.MODEL.SEQ2SEQ.INCL_DEC = True

""" math encoder """
_C.MODEL.MATH_ENC = CN()
_C.MODEL.MATH_ENC.DIM = 512
_C.MODEL.MATH_ENC.N_HEADS = 8
_C.MODEL.MATH_ENC.N_ENCODER_LAYERS = 6
_C.MODEL.MATH_ENC.N_DECODER_LAYERS = 6
_C.MODEL.MATH_ENC.DIM_FEEDFORWARD = 2048
_C.MODEL.MATH_ENC.DROPOUT = 0.0
_C.MODEL.MATH_ENC.SRC_VOCAB_SIZE = None
_C.MODEL.MATH_ENC.TGT_VOCAB_SIZE = None
_C.MODEL.MATH_ENC.SRC_SEQ_LEN = None
_C.MODEL.MATH_ENC.TGT_SEQ_LEN = None
_C.MODEL.MATH_ENC.INCL_DEC = False


# -----------------------------------------------------------------------------
# Best Model
# -----------------------------------------------------------------------------
_C.CKPT = CN()

""" Model """
_C.CKPT.DIR = "saved_models"
_C.CKPT.BEST = _C.CKPT.DIR + "/best.ckpt"
_C.CKPT.LAST = _C.CKPT.DIR + "/last.ckpt"


# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
_C.OPTIM = CN()
_C.OPTIM.NAME = None
_C.OPTIM.BASE_LR = 1e-4
_C.OPTIM.WARMUP_LR = 1e-7
_C.OPTIM.MIN_LR = 1e-6

""" SGD """
_C.OPTIM.SGD = CN()
_C.OPTIM.SGD.MOMENTUM = 0.90
_C.OPTIM.SGD.WEIGHT_DECAY = 1e-4
_C.OPTIM.SGD.NESTEROV = True

""" AdamW """
_C.OPTIM.ADAMW = CN()
_C.OPTIM.ADAMW.BETAS = (0.9, 0.999)
_C.OPTIM.ADAMW.EPS = 1e-8
_C.OPTIM.ADAMW.WEIGHT_DECAY = 1e-2


# -----------------------------------------------------------------------------
# Learning Rate Scheduler
# -----------------------------------------------------------------------------
_C.LRS = CN()
_C.LRS.NAME = None

""" CosineLRScheduler """
# set learning rate scheduler parameters in training
""" LinearLRScheduler """
# set learning rate scheduler parameters in training
""" StepLRScheduler """
_C.LRS.STEP_LRS = CN()
_C.LRS.STEP_LRS.DECAY_RATE = 0.1


# -----------------------------------------------------------------------------
# Criterion
# -----------------------------------------------------------------------------
_C.CRITERION = CN()
_C.CRITERION.NAME = None

""" CrossEntropy """
_C.CRITERION.CROSSENTROPY = CN()
_C.CRITERION.CROSSENTROPY.LABEL_SMOOTHING = 0.1

""" InfoNCE """
_C.CRITERION.INFONCE = CN()
_C.CRITERION.INFONCE.TEMPERATURE = 0.1
_C.CRITERION.INFONCE.REDUCTION = "mean"

""" SimCSE """
_C.CRITERION.SIMCSE = CN()
_C.CRITERION.SIMCSE.TEMPERATURE = 0.1
_C.CRITERION.SIMCSE.REDUCTION = "mean"

""" MaxSim """
_C.CRITERION.MAXSIM = CN()
_C.CRITERION.MAXSIM.TEMPERATURE = 0.1
_C.CRITERION.MAXSIM.REDUCTION = "mean"

""" Contrastive Loss """
_C.CRITERION.CONTRASTIVE = CN()
_C.CRITERION.CONTRASTIVE.MARGIN = 1.0
_C.CRITERION.CONTRASTIVE.REDUCTION = "mean"


# -----------------------------------------------------------------------------
# Postprocess
# -----------------------------------------------------------------------------
_C.POSTPROCESS = CN()
_C.POSTPROCESS.NAME = None


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.NAME = None
_C.DATA.EQUIV_PAIR = None
_C.DATA.VAL = None
_C.DATA.TEST = None
_C.DATA.CONTRASTIVE_EXPR = None
_C.DATA.N_EXPRS = None


# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------
_C.LOADER = CN()

""" Train DataLoader """
_C.LOADER.TRAIN = CN()
_C.LOADER.TRAIN.BATCH_SIZE = 256
_C.LOADER.TRAIN.SHUFFLE = True
_C.LOADER.TRAIN.NUM_WORKERS = 1
_C.LOADER.TRAIN.PIN_MEMORY = True

""" Val DataLoader """
_C.LOADER.VAL = CN()
_C.LOADER.VAL.BATCH_SIZE = 256
_C.LOADER.VAL.SHUFFLE = False
_C.LOADER.VAL.NUM_WORKERS = 1
_C.LOADER.VAL.PIN_MEMORY = True

""" Test DataLoader """
_C.LOADER.TEST = CN()
_C.LOADER.TEST.BATCH_SIZE = 256
_C.LOADER.TEST.SHUFFLE = False
_C.LOADER.TEST.NUM_WORKERS = 1
_C.LOADER.TEST.PIN_MEMORY = True


# -----------------------------------------------------------------------------
# Hyperparams
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(seed=SEED)
random.seed(a=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed_all(seed=SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
LOG_LEVEL = LogLevel.INFO


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

""" Training """
# epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.DECAY_EPOCHS = None
_C.TRAIN.MAX_NORM = 1.0
_C.TRAIN.N_ITER_PER_EPOCH = None
_C.TRAIN.WARMUP_EPOCHS = None
_C.TRAIN.N_EPOCHS = None
_C.TRAIN.SAVE_N_ITERS = None
_C.TRAIN.STATS_FILEPATH = "stats.json"


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()

""" Validation """
START = 25.0
END = 75.0
N = 3
TOL = 1e-10
SECS = 10


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print(f'[INFO] merge config from `{cfg_file}`')
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)
    _update_config_from_file(config, args.dataset)

    # cfg.defrost()
    # if args.opts:
    #     cfg.merge_from_list(args.opts)

    # cfg.freeze()


def get_config(args):
    """
    Get a yacs CfgNode object with default values.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    cfg = _C.clone()
    update_config(cfg, args)

    return cfg
