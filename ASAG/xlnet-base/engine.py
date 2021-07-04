from tqdm import tqdm
import torch.nn as nn
import torch




def loss_fn(outputs,target):
    x = torch.mean(outputs,dim=0,keepdim=True)
    y = torch.reshape(nn.functional.one_hot(target[0],6).float(),(1,6))
    return nn.BCEWithLogitsLoss()(x,y)


def train_fn(data_loader,model,optimizer,scheduler,device):
    model.train()

    for bi,d in tqdm(enumerate(data_loader),total=len(data_loader)):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets = d['target']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.long)


        optimizer.zero_grad()
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(outputs,targets)
        loss.backward()
        optimizer.step()
        scheduler.step()









def eval_fn(data_loader,model,device):
    model.eval()

    f_targets = []
    f_outputs = []

    with torch.no_grad():
        for bi,d in tqdm(enumerate(data_loader),total=len(data_loader)):
            ids = d['ids']
            token_type_ids = d['token_type_ids']
            mask = d['mask']
            targets = d['target']

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)


        
            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )

         
            
            m = torch.mean(outputs.float(),dim=0)
            outputs = torch.sigmoid(m)
            
            output = torch.tensor([torch.argmax(outputs)],dtype=torch.float)
            f_targets.extend(torch.tensor([targets[0].float()]).cpu().detach().numpy().tolist())
            f_outputs.extend(output.cpu().detach().numpy().tolist())
            
        return f_outputs,f_targets




