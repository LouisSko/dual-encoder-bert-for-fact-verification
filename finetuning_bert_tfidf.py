import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import sys
import os
import torch
from nlp_results import read_data, prepare_hover_dataset, prepare_tfidf_dataset
from bert_encoder_tfidf import *
from preprocessing import expand_rows, expand_retrieval_ids, map_retrieval_ids

# finetuning a dual encoder bert model on a dataset based on the top retrieval results of TF-iDF

def finetuning_bert_tfidf(batch_size=16, save_dir=None):

    # Set the device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    print('--- prepare training and validation data ---')
    df_data = prepare_tfidf_dataset(pd.read_json('/home/kit/stud/ulhni/nlp_project/hover-main/data/hover/tfidf_retrieved/train_tfidf_doc_retrieval_results.json'))

    # add the hpqa_id column
    df_hover = pd.read_json('/home/kit/stud/ulhni/nlp_project/hover-main/data/hover/hover_train_release_v1.1.json')
    df_data.insert(column = 'hpqa_id', loc=0, value = df_hover['hpqa_id'].copy())

    # train val split
    hpqa_id = list(set(df_hover['hpqa_id']))
    train_ids, test_ids = train_test_split(hpqa_id, train_size=0.9, shuffle=True, random_state=42)
    df_train = df_data[df_data['hpqa_id'].isin(train_ids)].reset_index(drop=True)
    df_valid = df_data[df_data['hpqa_id'].isin(test_ids)].reset_index(drop=True)
    n = 23

    df_train = expand_rows(df_train)
    df_train = expand_retrieval_ids(df_train, n)
    df_train = map_retrieval_ids(df_train)

    df_valid = expand_rows(df_valid)
    df_valid = expand_retrieval_ids(df_valid, n)
    df_valid = map_retrieval_ids(df_valid)
    print('--- training and validation data prepared ---')
    
    
    # create dataset and dataloader
    dataset_train = HoverDataset(df_train)
    dataset_valid = HoverDataset(df_valid)
    dataloader_train = DataLoader(dataset_train, batch_size, shuffle=True, collate_fn=custom_collate)
    dataloader_valid = DataLoader(dataset_valid, batch_size*4, shuffle=False, collate_fn=custom_collate) # bigger batch size possible for validation
    
    # intilize model
    model = DualEncoderBert('bert-base-uncased').to(device)
    model = nn.DataParallel(model)
    model.module.load_state_dict(torch.load('./training_results/bs200_no_dropout/best_model.pt'))
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.000005)
    
    # Criterion 
    criterion = nn.NLLLoss().to(device)

    # Training epochs
    num_epochs = 100

    print('--- start training ---')
    model = train(model, optimizer, criterion, dataloader_train, dataloader_valid, num_epochs, device, save=True, save_dir=save_dir)
    print('--- end training ---')
    
    
if __name__ == '__main__':
    # save_dir specifies the directory, where the model should be stored
    finetuning_bert_tfidf(batch_size=16, save_dir = '/home/kit/stud/ulhni/nlp_project/training_results/bs16_tfidf')