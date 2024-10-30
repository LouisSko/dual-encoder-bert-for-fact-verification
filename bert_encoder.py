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



class DualEncoderBert(nn.Module):
    """
    create dual encoder bert network
    """

    def __init__(self, pretrained_model_name):
        super().__init__()
        # Load the pre-trained BERT models for claims and docs
        self.claim_bert = BertModel.from_pretrained(pretrained_model_name)
        self.doc_bert = BertModel.from_pretrained(pretrained_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        # self.dropout = nn.Dropout(0.1)
        self.hidden_size = self.claim_bert.config.hidden_size
        # self.claim_pooling = nn.Linear(self.hidden_size, self.hidden_size)
        # self.doc_pooling = nn.Linear(self.hidden_size, self.hidden_size)

    def encode_claims(self, claim_input_ids, claim_attention_mask):
        # Encode claims using the claim-specific BERT model
        claim_outputs = self.claim_bert(input_ids=claim_input_ids, attention_mask=claim_attention_mask)[0]
        # claim_outputs = self.dropout(claim_outputs[:, :, :])  # Apply dropout
        return claim_outputs[:, 0, :]

    def encode_docs(self, doc_input_ids, doc_attention_mask):
        # Encode docs using the doc-specific BERT model
        doc_outputs = self.doc_bert(input_ids=doc_input_ids, attention_mask=doc_attention_mask)[0]
        # doc_outputs = self.dropout(doc_outputs[:, :, :])  # Apply dropout
        return doc_outputs[:, 0, :]

    def forward(self, claim_input_ids=None, claim_attention_mask=None, doc_input_ids=None, doc_attention_mask=None):

        claim_outputs, doc_outputs = None, None

        if claim_input_ids is not None and claim_attention_mask is not None:
            # Encode claims if claim inputs are provided
            claim_outputs = self.encode_claims(claim_input_ids, claim_attention_mask)

        if doc_input_ids is not None and doc_attention_mask is not None:
            # Encode docs if doc inputs are provided
            doc_outputs = self.encode_docs(doc_input_ids, doc_attention_mask)

        # claim_outputs, doc_outputs = pooling(claim_outputs, doc_outputs, pooling_typ='cls')

        
        # Normalize embeddings to unit length
        #if claim_outputs is not None:
        #    claim_outputs = F.normalize(claim_outputs, p=2, dim=1)

        #if doc_outputs is not None:
        #    doc_outputs = F.normalize(doc_outputs, p=2, dim=1)


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


class DualEncoderRoberta(nn.Module):
    """
    create dual encoder bert network
    """

    def __init__(self, pretrained_model_name):
        super().__init__()
        # Load the pre-trained roberta models for claims and docs

        self.claim_roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.doc_roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
      
        self.dropout = nn.Dropout(0.1)
        self.hidden_size = self.claim_roberta.config.hidden_size
        # self.claim_pooling = nn.Linear(self.hidden_size, self.hidden_size)
        # self.doc_pooling = nn.Linear(self.hidden_size, self.hidden_size)

    def encode_claims(self, claim_input_ids, claim_attention_mask):
        # Encode claims using the claim-specific roberta model
        claim_outputs = self.claim_roberta(input_ids=claim_input_ids, attention_mask=claim_attention_mask)[0]
        claim_outputs = self.dropout(claim_outputs[:, 0, :])  # Apply dropout
        # claim_outputs = F.normalize(claim_outputs, p=2, dim=1) # Normalize to unit length

        return claim_outputs

    def encode_docs(self, doc_input_ids, doc_attention_mask):
        # Encode docs using the doc-specific roberta model
        doc_outputs = self.doc_roberta(input_ids=doc_input_ids, attention_mask=doc_attention_mask)[0]
        doc_outputs = self.dropout(doc_outputs[:, 0, :])  # Apply dropout
        # doc_outputs = F.normalize(doc_outputs, p=2, dim=1) # Normalize to unit length

        return doc_outputs

    def forward(self, claim_input_ids=None, claim_attention_mask=None, doc_input_ids=None, doc_attention_mask=None):

        claim_outputs, doc_outputs = None, None

        if claim_input_ids is not None and claim_attention_mask is not None:
            # Encode claims if claim inputs are provided
            claim_outputs = self.encode_claims(claim_input_ids, claim_attention_mask)

        if doc_input_ids is not None and doc_attention_mask is not None:
            # Encode docs if doc inputs are provided
            doc_outputs = self.encode_docs(doc_input_ids, doc_attention_mask)

        # claim_outputs, doc_outputs = pooling(claim_outputs, doc_outputs, pooling_typ='cls')

        return claim_outputs, doc_outputs

    
class DataLoaderNegativeBatches(Dataset):
    """
    DataLoader which creates in batch negative samples randomly
    """

    def __init__(self, df_hover, batch_size, train=True):

        # df which contains columns: claim,supporting fact and text. One row for each supporting fact
        self.df = df_hover
        self.df = self.df.reset_index(drop=False)
        self.remaining_claim_docs = set(self.df.index) 
        self.train = train 
            
        self.bs = batch_size
        self.len = int(len(self.df) / self.bs)
        self.iterator = 0 
        
    def __len__(self):
        return self.len

    def is_end(self):
        if  len(self.remaining_claim_docs) == 0:
            self.remaining_claim_docs =  set(self.df.index)
            self.iterator = 0 
            return True

    def __iter__(self):
        while not self.is_end():
            self.iterator += 1 
            yield self.__getitem__(0)

    def __getitem__(self, index):
            
            
        # get n entries from but there should be no duplicate claims
        # get remaining data
        data_remaining = self.df[self.df.index.isin(self.remaining_claim_docs)]

        # get unique claims
        unique_claims = data_remaining['claim'].unique()

        # determine bs
        if len(unique_claims) < self.bs:
            bs = len(unique_claims)
        else:
            bs = self.bs

        if self.train: 
            # Sample n unique claims from the remaining unique claims
            sampled_claims = pd.Series(unique_claims).sample(n=bs, replace=False)
        
        if self.train is False:
            #for the validation set sample always the same claims
            sampled_claims = pd.Series(unique_claims).sample(n=bs, replace=False, random_state=42)
        
        
        # create a df with those sampled claims
        sampled_df = data_remaining[data_remaining['claim'].isin(sampled_claims)][['index','claim', 'text']]

        # remove all duplicates
        sampled_df = sampled_df.drop_duplicates('claim')

        # update claim_doc list
        self.remaining_claim_docs = set(self.remaining_claim_docs) - set(sampled_df['index']) 


        # generate labels
        labels = torch.arange(0, len(sampled_df))

        return sampled_df['claim'], sampled_df['text'], labels

    
class DataLoaderNegativeBatchesEmbeddingsMixed(Dataset):
    """
    DataLoader which creates in batch negatives. In every second batch it puts instances which have most similar embeddings.
    """

    def __init__(self, df_hover, batch_size, model, device):
        
        # df which contains columns: claim,supporting fact and text. One row for each supporting fact
        self.df = df_hover
        self.df = self.df.reset_index(drop=False)
        self.remaining_claim_docs = set(self.df.index)
        # map claim_doc combinations to idx (claims)
        self.mapping_claim_doc_idx = {index: name for index, name in self.df['idx'].items()}
            
        self.bs = batch_size
        self.len = int(len(self.remaining_claim_docs) / self.bs)
        self.device = device
        self.iterator = 0
        # create dataloader to prepare data to be encoded by the bert model
        self.ds_claims = TextDataGenerator(texts=self.df.drop_duplicates('idx').reset_index()['claim']) #df_hover is expanded. drop duplicates
        self.dl_claims = DataLoader(self.ds_claims, batch_size=64, shuffle=False)
        self.ds_docs = TextDataGenerator(texts=self.df['text'])
        self.dl_docs = DataLoader(self.ds_docs, batch_size=64, shuffle=False)
        
        #1024
        
        # initialize variables
        self.embeddings_claims = None
        self.embeddings_docs = None
        self.scores = None
        
        # calculate the embeddings
        self.update_embeddings(model)
        
            
    # calculate new embeddings at the end of epoch. need to be set in the code manually
    def update_embeddings(self, model):
        
        # encode docs and claims
        self.embeddings_claims = bert_encode_claims(model, self.dl_claims, self.device, None, save=False, ret=True)
        self.embeddings_docs = bert_encode_docs(model, self.dl_docs, self.device, None, save=False, ret=True)
    
    def __len__(self):
        return self.len

    def is_end(self):
        if  len(self.remaining_claim_docs) == 0:
            self.remaining_claim_docs =  set(self.df.index)
            self.iterator = 0 
            return True

    def __iter__(self):
        while not self.is_end():
            self.iterator += 1 
            yield self.__getitem__(0)

    
    def __getitem__(self, index):
        
        # get a batch with only similar documents
        if self.iterator % 2 == 0:
            
            # sample a random claim
            random_claim_doc = np.random.choice(list(self.remaining_claim_docs))
            
            # get the index of the claim in the embeddings
            idx_embeddings = self.df[self.df.index==random_claim_doc]['idx'].item()
            
            # calculate similarity
            self.scores = torch.matmul(self.embeddings_claims[idx_embeddings], self.embeddings_docs.T).to('cpu')
            
            # sort documents according to similarity
            docs_sorted_rel = torch.argsort(self.scores, descending = True)
            
            # get all the relevant documents for the claim (max 4 hops)
            positive_docs = set(self.df[self.df['idx']==idx_embeddings].index.values)

            # retrieve similar documents which are still in remaining_claim_docs, but not in positive_docs
            result = [claim_doc for claim_doc in docs_sorted_rel.tolist() if claim_doc in self.remaining_claim_docs and claim_doc not in positive_docs]

            # add sampled claim_doc combination
            claim_docs = (list([random_claim_doc]) + result)
            
            # remove duplicates. if two mappings are the same, remove the first one
            t = pd.DataFrame({'claim_docs':claim_docs})
            t['idx']=t['claim_docs'].apply(lambda x: self.mapping_claim_doc_idx[x])
            sampled_claim_docs = t.drop_duplicates('idx', keep='first')['claim_docs'][:self.bs]
            
            # create a df with the sampled claims
            sampled_df = self.df[self.df.index.isin(sampled_claim_docs)][['index','claim', 'text']]

            # update claim_doc list
            self.remaining_claim_docs = set(self.remaining_claim_docs) - set(sampled_claim_docs)


        # get a batch with random documents
        else:
            
            # get 32 entries from but there should be no duplicate claims
            # get remaining data
            data_remaining = self.df[self.df.index.isin(self.remaining_claim_docs)]
            
            # get unique claims
            unique_claims = data_remaining['claim'].unique()

            # determine bs
            if len(unique_claims) < self.bs:
                bs = len(unique_claims)
            else:
                bs = self.bs

            # Sample n unique claims from the remaining unique claims
            sampled_claims = pd.Series(unique_claims).sample(n=bs, replace=False)

            # create a df with those sampled claims
            sampled_df = data_remaining[data_remaining['claim'].isin(sampled_claims)][['index','claim', 'text']]

            # remove all duplicates
            sampled_df = sampled_df.drop_duplicates('claim')
            
            # update claim_doc list
            self.remaining_claim_docs = set(self.remaining_claim_docs) - set(sampled_df['index']) 
            
        
        # generate labels
        labels = torch.arange(0, len(sampled_df))
    
        
        return sampled_df['claim'], sampled_df['text'], labels
    

class HoverDataGenerator(Dataset):
    """
    create dataset for hover dataset
    """

    def __init__(self, claims, facts, labels):
        self.claim_list = claims
        self.fact_list = facts
        self.label_list = labels

    def __len__(self):
        return len(self.claim_list)

    def __getitem__(self, index):
        claim = self.claim_list[index]
        fact = self.fact_list[index]
        label = self.label_list[index]
        return claim, fact, label


class TextDataGenerator(Dataset):
    """
    create dataset for hover dataset
    """

    def __init__(self, texts):
        self.text_list = texts

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, index):
        text = self.text_list[index]

        return text

    
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

    
def pooling(enc_claim, enc_doc, pooling_typ):
    """
    Step 5: pooling to create a single vector from a document/claim
    :param pooling_typ: type of pooling that should be applied. 'mean' or 'cls'
    :param enc_claim: torch tensor with size [|claims|, 512, 768]
    :param enc_doc: torch tensor with size [|docs|, 512, 768]
    :return: reduced matrix to a single vector for each claim and doc
    """
    enc_claim_vec = None
    enc_doc_vec = None

    if pooling_typ == 'mean':
        # average over the dim=1
        if enc_claim is not None:
            enc_claim_vec = torch.mean(enc_claim, dim=1)
        if enc_doc is not None:
            enc_doc_vec = torch.mean(enc_doc, dim=1)

    if pooling_typ == 'cls':
        # only uses the cls token and discard all other information
        if enc_claim is not None:
            enc_claim_vec = enc_claim[:, 0, :]
        if enc_doc is not None:
            enc_doc_vec = enc_doc[:, 0, :]

    return enc_claim_vec, enc_doc_vec


def document_retrieval(enc_claims_vec, enc_docs_vec, top_k=2):
    """
    Step 6: Document Retrieval
    Perform Maximum Inner Product Search (MIPS). Return the top_k most relevant documents for a claim
    :param enc_docs_vec: encoded documents in the form [|docs|, 768]
    :param enc_claims_vec: encoded claims in the form [|claims|, 768]
    :param top_k:
    :return: scores and top_k documents
    """
    scores = torch.matmul(enc_claims_vec, enc_docs_vec.T)

    # in case the top indices should be returned

    top_indices = torch.topk(scores, k=top_k)[1]

    return scores, top_indices


def present_results(top_indices, doc_text):
    """
    Step 6: Present Result
    Rank the retrieved documents and present the results
    :param top_indices:
    :param doc_text:
    :return:
    """
    nr_of_claims = top_indices.shape[0]

    for claim in range(nr_of_claims):

        ranked_documents = [doc_text[int(idx)] for idx in top_indices[claim]]
        for rank, doc in enumerate(ranked_documents, start=1):
            print(f"Rank {rank}: {doc}")


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
        for batch_claims, batch_facts, batch_labels in tqdm(dataloader_train):
            claim_tokens = model.module.tokenize_text(batch_claims)
            fact_tokens = model.module.tokenize_text(batch_facts)

            claim_input_ids = claim_tokens['input_ids'].to(device)
            claim_attention_mask = claim_tokens['attention_mask'].to(device)
            doc_input_ids = fact_tokens['input_ids'].to(device)
            doc_attention_mask = fact_tokens['attention_mask'].to(device)
            targets = batch_labels.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.cuda.amp.autocast():
                
                claim_embedding, fact_embedding = model(claim_input_ids, claim_attention_mask, doc_input_ids,
                                                        doc_attention_mask)

                # calculate dot_product
                dot_product = torch.matmul(claim_embedding, fact_embedding.T)

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


# train model
def train_V2(model, optimizer, criterion, dataloader_train, dataloader_valid, num_epochs, device, save=True, save_dir=None):
    """
    same as train method, but it updates the dataloader every epoch
    """
    
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

    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        loss_list = []
        targets_all = torch.tensor([])
        preds_all = torch.tensor([])

        # Iterate over batches of data
        for batch_claims, batch_facts, batch_labels in tqdm(dataloader_train):
            claim_tokens = model.module.tokenize_text(batch_claims)
            fact_tokens = model.module.tokenize_text(batch_facts)

            claim_input_ids = claim_tokens['input_ids'].to(device)
            claim_attention_mask = claim_tokens['attention_mask'].to(device)
            doc_input_ids = fact_tokens['input_ids'].to(device)
            doc_attention_mask = fact_tokens['attention_mask'].to(device)
            targets = batch_labels.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            claim_embedding, fact_embedding = model(claim_input_ids, claim_attention_mask, doc_input_ids,
                                                    doc_attention_mask)

            # calculate dot_product
            dot_product = torch.matmul(claim_embedding, fact_embedding.T)

            # pass dot product through a softmax
            res_softmax = F.log_softmax(dot_product, dim=1)

            # calculate loss as negative log likelihood
            loss = criterion(res_softmax, targets)
            loss_list.append(loss.detach().to('cpu').numpy())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # store values for calculating the accuracy
            preds = torch.argmax(F.softmax(dot_product, dim=1), dim=1)
            targets_all = torch.cat([targets_all, targets.to('cpu')])
            preds_all = torch.cat([preds_all, preds.to('cpu')])
            
        # Clear memory after each epoch
        del claim_input_ids, claim_attention_mask, doc_input_ids, doc_attention_mask, targets, claim_embedding, fact_embedding
        torch.cuda.empty_cache()
        gc.collect()
        
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
        
         # save last model
        if save:
            torch.save(model.module.state_dict(), save_path_l_model)
            
        # implement early stopping and save best model
        if val_loss < best_val_loss - early_stopping_threshold:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model = model
            if save:
                torch.save(model.module.state_dict(), save_path_b_model)
            
            # update embeddings in dataloader. Leads to different batches
            dataloader_train.update_embeddings(best_model)
        
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

        for batch_claims, batch_facts, batch_labels in tqdm(dataloader):
            claim_tokens = model.module.tokenize_text(batch_claims)
            fact_tokens = model.module.tokenize_text(batch_facts)

            claim_input_ids = claim_tokens['input_ids'].to(device)
            claim_attention_mask = claim_tokens['attention_mask'].to(device)
            doc_input_ids = fact_tokens['input_ids'].to(device)
            doc_attention_mask = fact_tokens['attention_mask'].to(device)
            targets = batch_labels.to(device)
            
            with torch.cuda.amp.autocast():
                
                # Forward pass
                claim_embedding, fact_embedding = model(claim_input_ids, claim_attention_mask, doc_input_ids,
                                                        doc_attention_mask)

                # calculate dot_product
                dot_product = torch.matmul(claim_embedding, fact_embedding.T)

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


def bert_encode_claims(model, dataloader_test, device, save_dir=None, save = True, ret = False):
    
    if save_dir is None:
        save_dir = expanduser("~")  # Get the home directory
        
    os.makedirs(save_dir, exist_ok=True)
        
    model.eval()
    
    with torch.no_grad():
        # tensor to store embeddings
        all_pooled_claim_output = torch.empty(0, model.module.hidden_size)

        # Step 3: encode claims
        for batch in tqdm(dataloader_test):
            claim_tokens = model.module.tokenize_text(batch)
            pooled_claim_output, _ = model(claim_input_ids=claim_tokens['input_ids'].to(device),
                                           claim_attention_mask=claim_tokens['attention_mask'].to(device))
            all_pooled_claim_output = torch.cat([all_pooled_claim_output, pooled_claim_output.to('cpu')], dim=0)
        
        if save: 
            path = os.path.join(save_dir, 'embeddings_claims.pt')
            torch.save(all_pooled_claim_output, path)
            print(f'saved docs embeddings {path}')
        
        if ret:
            return all_pooled_claim_output
    
    
    
def bert_encode_docs(model, dataloader_db, device, save_dir=None, save = True, ret = False):
    
    if save_dir is None:
        save_dir = expanduser("~")  # Get the home directory
        
    os.makedirs(save_dir, exist_ok=True)
    
    it = 0
    model.eval()

    with torch.no_grad():
        # tensor to store embeddings
        all_pooled_doc_output = torch.empty(0, model.module.hidden_size)

        # Step 4: encode docs
        for batch in tqdm(dataloader_db):
            doc_tokens = model.module.tokenize_text(batch)
            _, pooled_doc_output = model(doc_input_ids=doc_tokens['input_ids'].to(device),
                                         doc_attention_mask=doc_tokens['attention_mask'].to(device))
            all_pooled_doc_output = torch.cat([all_pooled_doc_output, pooled_doc_output.to('cpu')], dim=0)

            # store data every n entries
            if len(all_pooled_doc_output) > 1_000_000:
                if save: 
                    torch.save(all_pooled_doc_output, os.path.join(save_dir, f'embeddings_wiki_{it}.pt'))
                all_pooled_doc_output = torch.empty(0, model.module.hidden_size)
                it += 1
                print('saved 1 mio docs')
        
        if save:        
            torch.save(all_pooled_doc_output, os.path.join(save_dir, f'embeddings_wiki_{it}.pt'))
            
        if ret:
            return all_pooled_doc_output