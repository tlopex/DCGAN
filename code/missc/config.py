from easydict import EasyDict as edict


__C = edict()
cfg = __C




__C.DISCRIMINATOR_LR = 0.0002
__C.GENERATOR_LR = 0.0002
__C.BATCH_SIZE = 32
__C.IMAGE_SIZE = 64
__C.CHANNELS_IMG = 3
__C.Z_DIM = 100
__C.NUM_EPOCH = 200
__C.FEATURES_D = 64
__C.FEATURES_G = 64