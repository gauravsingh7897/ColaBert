import config
import torch

class ColaDataset:
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
        self.max_len = config.MAX_LENGTH
        self.tokenizer = config.TOKENIZER

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        sentence = " ".join(sentence.split())

        encodings = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            pad_to_max_length=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'ids': torch.squeeze(encodings['input_ids'], 0),
            'masks': torch.squeeze(encodings['attention_mask'],0),
            'token_type_ids': torch.squeeze(encodings['token_type_ids'],0),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

    def __len__(self):
        return len(self.sentences)