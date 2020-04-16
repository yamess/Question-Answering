import torch
from transformers import BertTokenizer

MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 10
DROPOUT = 0.5
LR = 1e-3
TRAINING_STAT_PATH = "output/"
DEVICE = torch.device("cuda")

BERT_TOK_PATH = "bert-base-uncased"

BERT_MODEL_OUTPUT = "output/qa/bert/checkpoint.pt"

BERT_MODEL_PATH = "bert-large-uncased-whole-word-masking-finetuned-squad"

LOG_DIR = "output/log/"
OUTPUT_DIR = "output/"

