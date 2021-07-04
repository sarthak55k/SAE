import transformers
import config
import torch
import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased,self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.bert_drop = nn.Dropout(config.DROPOUT)
        self.out = nn.Linear(768,6)

    def forward(self,ids,mask,token_type_ids):
        _,o2 = self.bert(
            ids,
            attention_mask= mask,
            token_type_ids=token_type_ids

        )

        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output

