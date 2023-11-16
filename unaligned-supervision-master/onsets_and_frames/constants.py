import torch


SAMPLE_RATE = 16000
# HOP_LENGTH = 256
HOP_LENGTH = 512

ONSET_LENGTH = HOP_LENGTH
OFFSET_LENGTH = HOP_LENGTH

HOPS_IN_ONSET = ONSET_LENGTH // HOP_LENGTH
HOPS_IN_OFFSET = OFFSET_LENGTH // HOP_LENGTH
MIN_MIDI = 21
MAX_MIDI = 108
# MAX_MIDI = 127
N_KEYS = MAX_MIDI - MIN_MIDI + 1

DTW_FACTOR = 3

N_MELS = 229
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
WINDOW_LENGTH = 2048

SEQ_LEN = 327680

DRUM_CHANNEL = 9

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEFAULT_DEVICE = 'cpu'
