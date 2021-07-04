import config
import torch

class TransformerDataset:
    def __init__(self,dataframe,targets):
        super(TransformerDataset, self).__init__()
        self.tokenizer = config.TOKENIZER
        self.data = dataframe
        self.targets = targets
        self.max_length = config.MAX_LENGTH

    def __len__(self): 
        return(len(self.data))

    def __getitem__(self,index):
        sent1 = str(self.data.essay1[index])
        sent1 = " ".join(sent1.split())

        sent2 = str(self.data.essay2[index])
        sent2 = " ".join(sent2.split())

        inputs = self.tokenizer.encode_plus(
            sent1,
            sent2,
            add_special_tokens = True,
            max_length = self.max_length,
            truncation = True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        padding_length = self.max_length - len(ids)
        ids = ids + ([0]* padding_length)
        mask = mask + ([0]* padding_length)
        token_type_ids = token_type_ids + ([0]* padding_length)

        return{
            'ids': torch.tensor(ids,dtype = torch.long),
            'mask': torch.tensor(mask,dtype = torch.long),
            'token_type_ids': torch.tensor(token_type_ids,dtype = torch.long),
            'target': torch.tensor(self.targets[index],dtype=torch.float)
        }

