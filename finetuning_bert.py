import pandas as pd
from preprocessing import prepare_data
import torch
import torch.nn as nn
import torch.optim as optim
from bert_encoder import DualEncoderBert, DataLoaderNegativeBatches, train
import sys

def finetuning_bert(batch_size=32, save_dir=None):

    # Set the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('--- prepare training and validation data ---')
    # Read data and preprocesss
    df_hover = pd.read_json('/home/kit/stud/ulhni/nlp_project/hover-main/data/hover/hover_train_release_v1.1.json')
    df_hover_train, df_hover_valid = prepare_data(df_hover, train_size=0.9)
    print('--- training and validation data prepared ---')

    # initialize model
    model = DualEncoderBert('bert-base-uncased').to(device)
    model = nn.DataParallel(model)
    # model.module.load_state_dict(torch.load('./training_results/best_model.pt'))

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.00001)

    # Criterion (Loss Function) Negative Log Likelihood Loss
    criterion = nn.NLLLoss().to(device)

    # Training epochs
    num_epochs = 100

    # split in train and valid data
    dataloader_train = DataLoaderNegativeBatches(df_hover_train, batch_size, train=True)
    dataloader_valid = DataLoaderNegativeBatches(df_hover_valid, batch_size, train=False)

    # train model
    print('--- start training ---')
    model = train(model, optimizer, criterion, dataloader_train, dataloader_valid, num_epochs, device, save=True, save_dir=save_dir)
    print('--- end training ---')


if __name__ == '__main__':
    # save_dir specifies the directory, where the model should be stored
    finetuning_bert(batch_size=128, 
                    save_dir = '/home/kit/stud/ulhni/nlp_project/training_results/bs200_no_dropout')
