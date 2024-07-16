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

""" =============== Transformer =============== """
_C.MODEL.TX = CN()
_C.MODEL.TX.EMB_DIM = 512
# _C.MODEL.TX.SRC_VOCAB_SIZE = len(tokenizer.components)
# _C.MODEL.TX.TGT_VOCAB_SIZE = len(tokenizer.components)
_C.MODEL.TX.SRC_SEQ_LEN = 512
_C.MODEL.TX.TGT_SEQ_LEN = 256
_C.MODEL.TX.N_ENCODER_LAYERS = 6
_C.MODEL.TX.N_DECODER_LAYERS = 6
_C.MODEL.TX.N_HEADS = 8
_C.MODEL.TX.DROPOUT = 0.1
_C.MODEL.TX.DIM_FEEDFORWARD = 2048

""" AdamW """
_C.MODEL.TX.LR = 1e-4
_C.MODEL.TX.WEIGHT_DECAY = 1e-7


# -----------------------------------------------------------------------------
# Best Model
# -----------------------------------------------------------------------------
_C.BEST_MODEL = CN()

""" Model """
_C.BEST_MODEL.DIR = "models"
_C.BEST_MODEL.TX = _C.BEST_MODEL.DIR + "/tx.ckpt"


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
_C.DATA = CN()

""" EquivExpr """
_C.DATA.DATA_DIR = "data"
_C.DATA.TRAIN_FILE = _C.DATA.DATA_DIR + "/expr_triple_train.txt"


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
_C.LOADER.VAL.BATCH_SIZE = 512
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
_C.TRAIN.N_EPOCHS = 25
_C.TRAIN.LABEL_SMOOTHING = 0.1
_C.TRAIN.MAX_NORM = 1.0


# -----------------------------------------------------------------------------
# KMeans
# -----------------------------------------------------------------------------
_C.KMEANS = CN()

""" KMeans """
_C.KMEANS.MAX_ITER = 1000
_C.KMEANS.TOL = 1e-6
_C.KMEANS.RANDOM_STATE = SEED


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    # update_config(config, args)

    return config
