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
_C.MODEL.TX.DIM = 512
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
_C.CKPT = CN()

""" Model """
_C.CKPT.DIR = "models"
_C.CKPT.BEST = _C.CKPT.DIR + "/best.ckpt"
_C.CKPT.LAST = _C.CKPT.DIR + "/last.ckpt"


# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
_C.OPTIM = CN()

""" SGD """
_C.OPTIM.SGD = CN()
_C.OPTIM.SGD.MOMENTUM = 0.90
_C.OPTIM.SGD.WEIGHT_DECAY = 0.05
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

""" CosineLRScheduler """
# set learning rate scheduler parameters in training
""" LinearLRScheduler """
# set learning rate scheduler parameters in training
""" StepLRScheduler """
# set learning rate scheduler parameters in training


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
_C.DATA.TRAIN_FILE = _C.DATA.DATA_DIR + "/pairs.txt"
_C.DATA.VAL_FILE = _C.DATA.DATA_DIR + "/val.txt"


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

""" Optimizer """
_C.TRAIN.OPTIM = CN()
_C.TRAIN.OPTIM.NAME = "adamw"
_C.TRAIN.OPTIM.BASE_LR = 1e-4
_C.TRAIN.OPTIM.WARMUP_LR = 1e-7
_C.TRAIN.OPTIM.MIN_LR = 1e-6

""" LR Scheduler """
_C.TRAIN.LRS = CN()
_C.TRAIN.LRS.NAME = "cosine"
# epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LRS.DECAY_EPOCHS = 5
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LRS.DECAY_RATE = 0.1

""" Training """
_C.TRAIN.MAX_NORM = 1.0
_C.TRAIN.N_ITER_PER_EPOCH = 222674
_C.TRAIN.WARMUP_EPOCHS = 0.02245
_C.TRAIN.N_EPOCHS = 2
_C.TRAIN.SAVE_N_ITERS = 1000
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


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    # update_config(config, args)

    return config
