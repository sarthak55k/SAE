import transformers

import pandas as pd
MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 6
VALID_BATCH_SIZE = 6
TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
MODEL_PATH = '../output/model.bin'
DROPOUT = 0.3
LEARNING_RT = 3e-5
EPOCHS = 4

TRAINING_FILE = '../input/train_set.csv'
VALID_FILE = '../input/valid_set.csv'


