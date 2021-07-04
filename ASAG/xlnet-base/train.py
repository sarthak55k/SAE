import config
import engine
import dataset 
from model import Model
import pandas as pd 
import numpy as np
from sklearn import model_selection,metrics
from sklearn.metrics import f1_score
import torch
from transformers import get_linear_schedule_with_warmup, AdamW


def run():
    df_train = pd.read_csv(config.TRAINING_FILE)
    df_valid = pd.read_csv(config.VALID_FILE)

    


    train_dataset = dataset.TransformerDataset(
        dataframe=df_train,
        targets=df_train.score
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = config.TRAIN_BATCH_SIZE,
        num_workers = 4
    )

    valid_dataset = dataset.TransformerDataset(
        dataframe=df_valid,
        targets=df_valid.score
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = config.VALID_BATCH_SIZE,
        num_workers = 4
    )
    
    device = 'cuda'

    model = Model()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
    optimizer_parameters = [
        {
            "params": [
                p for n,p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weigth_decay": 0.001,
        },
        {
            "params": [
                p for n,p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weigth_decay": 0.0,
        }
    ]

    # Optimizer and Sceduler

    optimizer = torch.optim.AdamW(optimizer_parameters,config.LEARNING_RT)
    num_training_steps = int(len(train_dataset)/ config.TRAIN_BATCH_SIZE* config.EPOCHS)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    best_f1 = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader,model,optimizer,scheduler,device)
        # model.load_state_dict(torch.load('../output/model.bin'))
        outputs, targets = engine.eval_fn(valid_data_loader,model,device)
        accuracy = metrics.accuracy_score(targets,outputs)
        f1 = f1_score(targets,outputs,average='micro')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1_score = {f1}")
        if f1 > best_f1:
            torch.save(model.state_dict(),config.MODEL_PATH + '.pt')
            best_accuracy = accuracy
    



if __name__ == "__main__":
    run()
