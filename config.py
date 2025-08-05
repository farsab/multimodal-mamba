BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 10
NUM_CLASSES = 4
SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
