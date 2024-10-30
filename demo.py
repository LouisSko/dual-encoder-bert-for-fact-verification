import sqlite3
import pandas as pd
import torch
import torch.nn as nn
from bert_encoder import DualEncoderBert, TextDataGenerator, bert_encode_claims, bert_encode_docs
from torch.utils.data import DataLoader
import pickle
import unicodedata
import pprint
import faiss
from transformers import logging

def create_faiss_index(model_name):

    embeddings_wiki_0 = torch.load(f'/home/kit/stud/ulhni/nlp_project/embeddings/{model_name}/embeddings_wiki_0.pt')
    embeddings_wiki_1 = torch.load(f'/home/kit/stud/ulhni/nlp_project/embeddings/{model_name}/embeddings_wiki_1.pt')
    embeddings_wiki_2 = torch.load(f'/home/kit/stud/ulhni/nlp_project/embeddings/{model_name}/embeddings_wiki_2.pt')
    embeddings_wiki_3 = torch.load(f'/home/kit/stud/ulhni/nlp_project/embeddings/{model_name}/embeddings_wiki_3.pt')
    embeddings_wiki_4 = torch.load(f'/home/kit/stud/ulhni/nlp_project/embeddings/{model_name}/embeddings_wiki_4.pt')
    embeddings_wiki_5 = torch.load(f'/home/kit/stud/ulhni/nlp_project/embeddings/{model_name}/embeddings_wiki_5.pt')

    embeddings_wiki = torch.cat([embeddings_wiki_0, 
                                 embeddings_wiki_1, 
                                 embeddings_wiki_2, 
                                 embeddings_wiki_3, 
                                 embeddings_wiki_4, 
                                 embeddings_wiki_5])

    del embeddings_wiki_0, embeddings_wiki_1, embeddings_wiki_2, embeddings_wiki_3, embeddings_wiki_4, embeddings_wiki_5

    # Convert tensor to numpy array
    embeddings_wiki_np = embeddings_wiki.numpy()

    # Initialize FAISS index. The dimension of the vectors is given as an argument
    index = faiss.IndexFlatIP(embeddings_wiki_np.shape[1])

    # Add vectors to the index
    index.add(embeddings_wiki_np)

    # Save the index to disk
    faiss.write_index(index, 'wiki_embeddings.index')

    
logging.set_verbosity_error()


model_name = 'bs128'
# check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize model
model = DualEncoderBert('bert-base-uncased').to(device)
model = nn.DataParallel(model)
# load finetuned model
path_model=f'/home/kit/stud/ulhni/nlp_project/training_results/{model_name}/best_model.pt'
model.module.load_state_dict(torch.load(path_model, map_location=torch.device(device)))

# read mapping of articles to index
file_path = './article_name_index_map.pkl'
with open(file_path, 'rb') as file:
    article_name_index_map = pickle.load(file)
    
article_index_name_map = {value: key for key, value in article_name_index_map.items()}

# Load the index from disk
index = faiss.read_index('wiki_embeddings.index')


def retrieve_articles(text, top_k=5):

    claim_tokens = model.module.tokenize_text([text])
    claim_output, _ = model(claim_input_ids=claim_tokens['input_ids'].to(device), claim_attention_mask=claim_tokens['attention_mask'].to(device))

    # Perform the search
    scores, indices = index.search(claim_output.detach().numpy(), k=top_k)

    # Use map function
    series = pd.Series(indices[0])
    mapped_series = series.map(article_index_name_map)
    
    # retrieve article text
    results = {}
    conn = sqlite3.connect('/home/kit/stud/ulhni/nlp_project/hover-main/data/wiki_wo_links.db')
    cursor = conn.cursor()
    for i, title in enumerate(mapped_series):
        cursor.execute("SELECT text FROM documents WHERE id = ?", (unicodedata.normalize('NFD', title),))
        result = cursor.fetchall()
        results[title] = result[0][0]

    conn.close()
    
    for key, value in results.items():
        print(f"\n----------\n\n{key}:\n\n{value}")