import sqlite3
import pandas as pd
import torch
import torch.nn as nn
from bert_encoder import DualEncoderBert, TextDataGenerator, bert_encode_claims, bert_encode_docs
from torch.utils.data import DataLoader


def encode(path_model, batch_size=512, save_dir=None):
    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read test data
    df_hover_test = pd.read_json('/home/kit/stud/ulhni/nlp_project/hover-main/data/hover/hover_dev_release_v1.1.json')

    # initialize model
    model = DualEncoderBert('bert-base-uncased').to(device)
    model = nn.DataParallel(model)

    # load finetuned model
    model.module.load_state_dict(torch.load(path_model))

    # encode claims using bert model
    print('--- start encoding claims ---')
    dataset_test = TextDataGenerator(texts=df_hover_test['claim'])
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    bert_encode_claims(model, dataloader_test, device, save_dir=save_dir, save=True, ret=False)

    # read database data
    conn = sqlite3.connect('/home/kit/stud/ulhni/nlp_project/hover-main/data/wiki_wo_links.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents")
    result = cursor.fetchall()
    conn.close()
    df_db = pd.DataFrame(result, columns=['article_name', 'text'])
    df_db['text'] = df_db['article_name'] + '[SEP] ' + df_db['text']

    # encode docs using bert model
    print('--- start encoding docs ---')
    dataset_db = TextDataGenerator(texts=df_db['text'])
    dataloader_db = DataLoader(dataset_db, batch_size=batch_size, shuffle=False)
    bert_encode_docs(model, dataloader_db, device, save_dir=save_dir, save=True, ret=False)


if __name__ == '__main__':
    encode(path_model='/home/kit/stud/ulhni/nlp_project/training_results/bs16_tfidf/best_model.pt', 
           batch_size=1024, 
           save_dir = '/home/kit/stud/ulhni/nlp_project/embeddings/bs16_tfidf')

    #v3 is mixedembeddings dataloader