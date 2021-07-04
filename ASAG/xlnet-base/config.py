import transformers
import torch

SEED = 42
MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 6
VALID_BATCH_SIZE = 6
LABELS = 6



#data
TOKENIZER = transformers.XLNetTokenizer.from_pretrained('xlnet-base-cased')
MODEL_PATH = '/kaggle/working/xlnet-best'
DROPOUT = 0.3
LEARNING_RT = 3e-5
EPOCHS = 2

TRAINING_FILE = '../data/train_set.csv'
VALID_FILE = '../data/valid_set.csv'


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FULL_FINETUNING = True
SAVE_BEST_ONLY = True


