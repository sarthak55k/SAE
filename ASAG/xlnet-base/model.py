import transformers
import config
import torch
import torch.nn as nn


# Sentence_A + [SEP] + Sentence_B + [SEP] + [CLS]

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.trans_model = transformers.XLNetForSequenceClassification.from_pretrained(
            'xlnet-base-cased',
            num_labels = config.LABELS
            )
        # self.xlnet_drop = nn.Dropout(config.DROPOUT)
        # self.out = nn.Linear(768,6)

    def forward(self,ids,mask,token_type_ids):
        logits =  self.trans_model(
            ids,
            attention_mask= mask,
            token_type_ids=token_type_ids

        )

        return logits[0]

