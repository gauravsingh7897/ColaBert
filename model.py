import config
import transformers
import torch.nn as nn


class ColaBert(nn.Module):
    def __init__(self):
        super(ColaBert,self).__init__()
        self.bert = transformers.BertForSequenceClassification.from_pretrained(config.BERT_PATH, num_labels=2, output_attentions=False, output_hidden_states=False)

    def forward(self, ids, masks, labels):
        outputs = self.bert(
            ids, token_type_ids=None, attention_mask=masks, labels=labels)
        return outputs[0], outputs[1]
