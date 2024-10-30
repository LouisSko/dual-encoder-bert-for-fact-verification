from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torch_lr_finder import LRFinder
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from sklearn import metrics
import os
from os.path import expanduser
import gc


# this is a separate approach where a dataset is created based on the results of the tfidf. It differs from bert_encoder due to a different train method

class DualEncoderBert(nn.Module):
    """
    create dual encoder bert network
    """

    def __init__(self, pretrained_model_name):
        super().__init__()
        # Load the pre-trained bert models for claims and docs

        self.claim_bert = BertModel.from_pretrained(pretrained_model_name)
        self.doc_bert = BertModel.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.hidden_size = self.claim_bert.config.hidden_size

    def encode_claims(self, claim_input_ids, claim_attention_mask):
        # Encode claims using the claim-specific bert model
        claim_outputs = self.claim_bert(input_ids=claim_input_ids, attention_mask=claim_attention_mask)[0] # get the hidden states 
        claim_outputs = claim_outputs[:, 0, :]  # get the cls token
        return claim_outputs

    def encode_docs(self, doc_input_ids, doc_attention_mask):
        # Encode docs using the doc-specific bert model
        doc_outputs = self.doc_bert(input_ids=doc_input_ids, attention_mask=doc_attention_mask)[0] # get the hidden states 
        doc_outputs = doc_outputs[:, 0, :]  # get the cls token
        return doc_outputs

    def forward(self, claim_input_ids=None, claim_attention_mask=None, doc_input_ids=None, doc_attention_mask=None):

        claim_outputs, doc_outputs = None, None

        if claim_input_ids is not None and claim_attention_mask is not None:
            # Encode claims if claim inputs are provided
            claim_outputs = self.encode_claims(claim_input_ids, claim_attention_mask)

        if doc_input_ids is not None and doc_attention_mask is not None:
            # Encode docs if doc inputs are provided
            doc_outputs = self.encode_docs(doc_input_ids, doc_attention_mask)


        return claim_outputs, doc_outputs

    
    def tokenize_text(self, texts):
        """
        tokenizes claims and texts
        :param texts: text as a list
        :return:
        """
        text_tokens = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return text_tokens

    
class HoverDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Fetch the data item at the specified index
        item = self.data.iloc[index]
        
        claim = item['claim']
        facts = list(item['supporting_fact_id':])
        
        # label = np.zeros(len(facts))  # Create an array of zeros with length n
        # label[0] = 1  # Set the first element to 1
        
        return claim, facts


# for changing the behaviour of pytorch dataloader
def custom_collate(batch):
    claims = [item[0] for item in batch]
    facts = [item[1] for item in batch]

    return claims, facts


# train model
def train(model, optimizer, criterion, dataloader_train, dataloader_valid, num_epochs, device, save=True, save_dir=None):
    
    # specify paths and create dicts
    if save_dir is None:
        save_dir = expanduser("~")  # Get the home directory
        
    os.makedirs(save_dir, exist_ok=True)
    save_path_b_model = os.path.join(save_dir, 'best_model.pt')
    save_path_l_model = os.path.join(save_dir, 'last_model.pt')
    save_path_results = os.path.join(save_dir, 'results.csv')
    
    best_val_loss = float('inf')
    best_epoch = 0
    best_model = None
    early_stopping_threshold = 1e-8  # Define a threshold for the difference in metric between consecutive epochs
    patience = 15  # define a patience parameter
    df_results = pd.DataFrame(columns=['train_loss', 'train_acc', 'val_loss', 'val_acc'])
    
    # import gradscaler
    scaler = torch.cuda.amp.GradScaler()
        
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        loss_list = []
        targets_all = torch.tensor([])
        preds_all = torch.tensor([])

        # Iterate over batches of data
        for i, (batch_claims, batch_facts) in enumerate(tqdm(dataloader_train)):
            
            model.train()
            
            # tokenize claims and facts
            batch_facts = sum(batch_facts, [])

            # encode claims and docs
            claim_tokens = model.module.tokenize_text(batch_claims)
            fact_tokens = model.module.tokenize_text(batch_facts)

            # extract input_ids and attention_mask
            claim_input_ids = claim_tokens['input_ids'].to(device)
            claim_attention_mask = claim_tokens['attention_mask'].to(device)
            doc_input_ids = fact_tokens['input_ids'].to(device)
            doc_attention_mask = fact_tokens['attention_mask'].to(device)

            # get targets/labels. It's always the first document thats relevant
            targets = torch.zeros(len(claim_input_ids), dtype=torch.long).to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.cuda.amp.autocast():

                claim_embedding, fact_embedding = model(claim_input_ids,
                                                        claim_attention_mask,
                                                        doc_input_ids,
                                                        doc_attention_mask
                                                        )
                # calculate dot_product
                dot_product = []
                step_size = int(len(fact_embedding)/len(claim_embedding))
                num_iterations = int(len(claim_embedding))

                for it in range(num_iterations):
                    start_index = it * step_size
                    end_index = start_index + step_size
                    result = torch.matmul(claim_embedding[it], fact_embedding[start_index:end_index].T) # TODO: hier noch claim embedding mit an fact_embedding dranhängen
                    dot_product.append(result)

                # concatenate all results
                dot_product = torch.stack(dot_product)

                # pass dot product through a softmax
                res_softmax = F.log_softmax(dot_product, dim=1)

                # calculate loss as negative log likelihood
                loss = criterion(res_softmax, targets)
            
            # Backward pass and optimization with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_list.append(loss.detach().to('cpu').numpy())
                
            # store values for calculating the accuracy
            preds = torch.argmax(F.softmax(dot_product, dim=1), dim=1)
            targets_all = torch.cat([targets_all, targets.to('cpu')])
            preds_all = torch.cat([preds_all, preds.to('cpu')])
            
            # Calculate accuracy for the epoch
            if (i % 500 == 0) and (i != 0):
                # save model during epoch because one epoch takes 3 hours
                if save:
                    torch.save(model.module.state_dict(), save_path_l_model)

                acc_rolling = metrics.accuracy_score(targets_all[-500:], preds_all[-500:])
                loss_rolling = np.mean(loss_list[-500:])
                print(f'Current batch: {i}. Average over last 500 Batches: Loss: {loss_rolling}. Average accuary is {acc_rolling}')
        
        # Clear memory after each epoch
        del claim_input_ids, claim_attention_mask, doc_input_ids, doc_attention_mask, targets, claim_embedding, fact_embedding
        torch.cuda.empty_cache()
        gc.collect()
        
        # save last model
        if save:
            torch.save(model.module.state_dict(), save_path_l_model)
    
        # Calculate average loss for the epoch
        train_loss = np.mean(loss_list)
        
        # Calculate accuracy for the epoch
        train_acc = metrics.accuracy_score(targets_all, preds_all)
        
        # validation and early stopping
        val_loss, val_acc = validate(model, criterion, dataloader_valid, device)

        # save results
        df_results.loc[
            epoch+1, ['train_loss', 'train_acc', 'val_loss', 'val_acc']] = train_loss, train_acc, val_loss, val_acc
        if save:
            df_results.to_csv(save_path_results)

        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs} - Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f} ")
        

        # implement early stopping and save best model
        if val_loss < best_val_loss - early_stopping_threshold:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model = model
            if save:
                torch.save(best_model.module.state_dict(), save_path_b_model)
        
        # If the metric has not improved for the last `n` epochs Stop training
        elif epoch - best_epoch > patience:
            print(f'Early Stopping! best epoch: {best_epoch + 1}')
            break

    return best_model


def validate(model, criterion, dataloader, device):
    """
        function to validate the model
    """
    model.eval()

    with torch.no_grad():  # don't calculate gradients in validation

        loss_list = []
        # Initialize an empty tensor for targets_all and preds_all
        targets_all = torch.tensor([])
        preds_all = torch.tensor([])

        for batch_claims, batch_facts in tqdm(dataloader):
            
            # tokenize claims and facts
            batch_facts = sum(batch_facts, [])

            # encode claims and docs
            claim_tokens = model.module.tokenize_text(batch_claims)
            fact_tokens = model.module.tokenize_text(batch_facts)

            # extract input_ids and attention_mask
            claim_input_ids = claim_tokens['input_ids'].to(device)
            claim_attention_mask = claim_tokens['attention_mask'].to(device)
            doc_input_ids = fact_tokens['input_ids'].to(device)
            doc_attention_mask = fact_tokens['attention_mask'].to(device)

            # get targets/labels. It's always the first document thats relevant
            targets = torch.zeros(len(claim_input_ids), dtype=torch.long).to(device)
            
            with torch.cuda.amp.autocast():
                
                claim_embedding, fact_embedding = model(claim_input_ids,
                                                        claim_attention_mask,
                                                        doc_input_ids,
                                                        doc_attention_mask
                                                        )
                # calculate dot_product
                dot_product = []
                step_size = int(len(fact_embedding)/len(claim_embedding))
                num_iterations = int(len(claim_embedding))

                for it in range(num_iterations):
                    start_index = it * step_size
                    end_index = start_index + step_size
                    result = torch.matmul(claim_embedding[it], fact_embedding[start_index:end_index].T)
                    dot_product.append(result)

                # concatenate all results
                dot_product = torch.stack(dot_product)

                # pass dot product through a softmax
                res_softmax = F.log_softmax(dot_product, dim=1)

                # calculate loss as negative log likelihood
                loss = criterion(res_softmax, targets)
            
                
            loss_list.append(loss.detach().to('cpu').numpy())

            # store values for calculating the accuracy
            preds = torch.argmax(F.softmax(dot_product, dim=1), dim=1)
            targets_all = torch.cat([targets_all, targets.to('cpu')])
            preds_all = torch.cat([preds_all, preds.to('cpu')])
            
        # Clear memory after each epoch
        del claim_input_ids, claim_attention_mask, doc_input_ids, doc_attention_mask, targets, claim_embedding, fact_embedding
        torch.cuda.empty_cache()
        gc.collect()
            
        # Calculate average loss for the epoch
        avg_loss = np.mean(loss_list)

        # Calculate accuracy
        acc = metrics.accuracy_score(targets_all, preds_all)

    return avg_loss, acc



