import transformers


MAX_LENGTH = 64
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 10

BERT_PATH = "./data/bert_base_uncased/"
MODEL_PATH = "model.bin"

TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH, do_lower_case=True)
