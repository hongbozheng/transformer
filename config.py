import torch
from logger import LogLevel
from yacs.config import CfgNode as CN


_C = CN()

# Base config files
_C.BASE = ['']


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()

""" Transformer """
_C.MODEL.TX = CN()
_C.MODEL.TX.EMB_DIM = 512
# _C.MODEL.TX.SRC_VOCAB_SIZE = len(tokenizer.components)
# _C.MODEL.TX.TGT_VOCAB_SIZE = len(tokenizer.components)
_C.MODEL.TX.SRC_SEQ_LEN = 200
_C.MODEL.TX.TGT_SEQ_LEN = 200
_C.MODEL.TX.N_ENCODER_LAYERS = 6
_C.MODEL.TX.N_DECODER_LAYERS = 6
_C.MODEL.TX.N_HEADS = 8
_C.MODEL.TX.DROPOUT = 0.1
_C.MODEL.TX.DIM_FEEDFORWARD = 2048


# -----------------------------------------------------------------------------
# Best Model
# -----------------------------------------------------------------------------
_C.BEST_MODEL = CN()

""" Model """
_C.BEST_MODEL.DIR = "models"
_C.BEST_MODEL.TX = _C.BEST_MODEL.DIR + "/tx.ckpt"


# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
_C.OPTIM = CN()

""" AdamW """
_C.OPTIM.ADAMW = CN()
_C.OPTIM.ADAMW.LR = 1e-4
_C.OPTIM.ADAMW.WEIGHT_DECAY = 1e-2


# -----------------------------------------------------------------------------
# Learning Rate Scheduler
# -----------------------------------------------------------------------------
_C.LRS = CN()

""" CosineAnnealingWarmRestarts """
_C.LRS.CAWR = CN()
_C.LRS.CAWR.T_0 = 10
_C.LRS.CAWR.T_MULT = 2
_C.LRS.CAWR.ETA_MIN = 1e-8
_C.LRS.CAWR.LAST_EPOCH = -1

""" CosineAnnealingLR """
_C.LRS.CALR = CN()
_C.LRS.CALR.T_MAX = 50
_C.LRS.CALR.ETA_MIN = 1e-8
_C.LRS.CALR.LAST_EPOCH = -1


# -----------------------------------------------------------------------------
# Criterion
# -----------------------------------------------------------------------------
_C.CRITERION = CN()

""" CrossEntropy """
_C.CRITERION.CROSSENTROPY = CN()
_C.CRITERION.CROSSENTROPY.LABEL_SMOOTHING = 0.1


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.DATA = CN()

""" EquivExpr """
_C.DATA.DATA_DIR = "data"
_C.DATA.TRAIN_FILE = _C.DATA.DATA_DIR + "/expr_pairs.txt"
_C.DATA.VAL_FILE = _C.DATA.DATA_DIR + "/exprs_val.txt"


# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------
_C.LOADER = CN()

""" Train DataLoader """
_C.LOADER.TRAIN = CN()
_C.LOADER.TRAIN.BATCH_SIZE = 256
_C.LOADER.TRAIN.SHUFFLE = False
_C.LOADER.TRAIN.NUM_WORKERS = 1
_C.LOADER.TRAIN.PIN_MEMORY = True

""" Val DataLoader """
_C.LOADER.VAL = CN()
_C.LOADER.VAL.BATCH_SIZE = 256
_C.LOADER.VAL.SHUFFLE = False
_C.LOADER.VAL.NUM_WORKERS = 1
_C.LOADER.VAL.PIN_MEMORY = True


# -----------------------------------------------------------------------------
# Hyperparams
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
_C.TRAIN.N_EPOCHS = 50
_C.TRAIN.MAX_NORM = 4.0


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


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    # update_config(config, args)

    return config
