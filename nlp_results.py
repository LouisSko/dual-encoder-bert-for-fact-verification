import numpy as np
import torch
from tqdm import tqdm
import os 
import unicodedata
import pandas as pd 
import pickle 
import sqlite3
import math
import matplotlib.ticker as mtick


SEP_TOKEN = ' [SEP] '
#SEP_TOKEN = '. '


def read_data(model):
    
    embeddings_claims = torch.load(f'/home/kit/stud/ulhni/nlp_project/embeddings/{model}/embeddings_claims.pt')

    embeddings_wiki_0 = torch.load(f'/home/kit/stud/ulhni/nlp_project/embeddings/{model}/embeddings_wiki_0.pt')
    embeddings_wiki_1 = torch.load(f'/home/kit/stud/ulhni/nlp_project/embeddings/{model}/embeddings_wiki_1.pt')
    embeddings_wiki_2 = torch.load(f'/home/kit/stud/ulhni/nlp_project/embeddings/{model}/embeddings_wiki_2.pt')
    embeddings_wiki_3 = torch.load(f'/home/kit/stud/ulhni/nlp_project/embeddings/{model}/embeddings_wiki_3.pt')
    embeddings_wiki_4 = torch.load(f'/home/kit/stud/ulhni/nlp_project/embeddings/{model}/embeddings_wiki_4.pt')
    embeddings_wiki_5 = torch.load(f'/home/kit/stud/ulhni/nlp_project/embeddings/{model}/embeddings_wiki_5.pt')

    embeddings_wiki = torch.cat([embeddings_wiki_0, 
                                 embeddings_wiki_1, 
                                 embeddings_wiki_2, 
                                 embeddings_wiki_3, 
                                 embeddings_wiki_4, 
                                 embeddings_wiki_5])

    del embeddings_wiki_0, embeddings_wiki_1, embeddings_wiki_2, embeddings_wiki_3, embeddings_wiki_4, embeddings_wiki_5

    return embeddings_claims, embeddings_wiki


# add retrieval ids
def add_retrieval_ids(df_hover, mips_top_indices):
    df_hover['retrieval_ids'] = None
    df_hover['retrieval_ids'] = df_hover['retrieval_ids'].astype('object')
    for i in np.arange(df_hover.shape[0]):
        df_hover.at[i, 'retrieval_ids']=mips_top_indices[i].tolist()
        
    return df_hover

# create dictionary which maps article names to their corresponding indexes
def load_wiki_DB(create_mapping=True):
    conn = sqlite3.connect('hover-main/data/wiki_wo_links.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents")
    result = cursor.fetchall()
    conn.close()
    df_DB = pd.DataFrame(result, columns=['article_name', 'text'])
    
    if create_mapping:
        article_name_index_map = {name: index for index, name in df_DB['article_name'].items()}
        # Save the dictionary as a pickle file
        file_path = './article_name_index_map.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(article_name_index_map, file)
        return df_DB, article_name_index_map
    
    return df_DB, article_name_index_map

# Perform maximum inner product search
def mips(embeddings_claims, embeddings_wiki, model, top_k = 100, save=True):
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Number of top-k results to retrieve

    # Perform the search
    results_top_indices = torch.empty([0,top_k])
    results_top_scores = torch.empty([0,top_k])
    batch_size = 128
    length = embeddings_claims.shape[0]
    for i in tqdm(range(0,math.ceil(length/batch_size))):
        start = batch_size * i
        end = batch_size * (i + 1)

        # Perform Maximum Inner Product Search
        scores = torch.matmul(embeddings_claims[start:end].to(device), embeddings_wiki.to(device).T)
        top_scores, top_indices = torch.topk(scores, k=top_k)

        results_top_indices = torch.cat([results_top_indices, top_indices.to('cpu')])
        results_top_scores = torch.cat([results_top_scores, top_scores.to('cpu')])

    # save the results
    torch.save(results_top_indices, os.path.join(f'./embeddings/{model}', f'mips_top_{top_k}.pt'))
    torch.save(results_top_scores, os.path.join(f'./embeddings/{model}', f'mips_scores_{top_k}.pt'))
    
    return results_top_indices, results_top_scores

# load results from mips
def load_mips_results(model, top_k):
    
    mips_top_indices = torch.load(os.path.join(f'./embeddings/{model}', f'mips_top_{top_k}.pt'))
    mips_top_scores = torch.load(os.path.join(f'./embeddings/{model}', f'mips_scores_{top_k}.pt'))
    
    return mips_top_indices, mips_top_scores

    
def load_article_mapping():
    # load a dictionary which maps article names to their corresponding indexes
    file_path = './article_name_index_map.pkl'
    with open(file_path, 'rb') as file:
        article_name_index_map = pickle.load(file)
        
    return article_name_index_map


# add corresponding ids in the DB for the supporting facts  
def prepare_hover_dataset(df):
    
    article_name_index_map = load_article_mapping()
    
    df['num_hops'] = df['num_hops'].astype(int)
    df['supporting_facts_list'] = None
    df['supporting_facts_ids'] = None

    for idx, facts in enumerate(tqdm(df['supporting_facts'])):
        claim = df.loc[idx, 'claim']
        label = df.loc[idx, 'label']
        fact_list = []
        fact_id = []

        for fact in facts:
            normalized_fact = unicodedata.normalize('NFD', fact[0])
            fact_list.append(normalized_fact)
            fact_id.append(article_name_index_map[normalized_fact])

        df.at[idx,'supporting_facts_list'] = list(set(fact_list))
        df.at[idx,'supporting_facts_ids'] = list(set(fact_id))
        
    return df


# add corresponding ids in the DB for the supporting facts
def prepare_tfidf_dataset(df):
    
    article_name_index_map = load_article_mapping()
    
    df['supporting_facts_list'] = None
    df['supporting_facts_ids'] = None
    df['retrieval_list'] = None
    df['retrieval_ids'] = None

    # extract evidence
    for idx, facts in enumerate(tqdm(df['evidence'])):
        claim = df.loc[idx, 'claim']
        label = df.loc[idx, 'label']
        fact_list = []
        fact_id = []

        for fact in facts[0]:
            normalized_fact = unicodedata.normalize('NFD', fact[2])
            fact_list.append(normalized_fact)
            fact_id.append(article_name_index_map[normalized_fact])

        df.at[idx,'supporting_facts_list'] = list(set(fact_list))
        df.at[idx,'supporting_facts_ids'] = list(set(fact_id))

    # extract retrieval results
    for idx, facts in enumerate(tqdm(df['doc_retrieval_results'])):
        claim = df.loc[idx, 'claim']
        label = df.loc[idx, 'label']
        fact_list = []
        fact_id = []

        for fact in facts[0][0]:
            normalized_fact = unicodedata.normalize('NFD', fact)
            fact_list.append(normalized_fact)
            fact_id.append(article_name_index_map[normalized_fact])

        df.at[idx,'retrieval_list'] = fact_list
        df.at[idx,'retrieval_ids'] = fact_id
        
    return df


def calculate_matches(df, top_k):
    
    # get matches and hops
    for i in df.index:
        df.loc[i, 'number_of_matches'] = len(set(df.loc[i,'supporting_facts_ids']) & set(df.loc[i,'retrieval_ids'][:top_k]))
        df.loc[i, 'num_hops'] = len(df.loc[i,'supporting_facts_ids'])

    return df

def calculate_metrics(df, top_k_list, num_hops_list, label = 'all'):

    df_recall = pd.DataFrame(columns=top_k_list, index = num_hops_list)
    # relative anzahl an vollständigen übereinstimmungen. z.B. 2/2 hops für einen Datenpunkt
    df_hits = pd.DataFrame(columns=top_k_list, index = num_hops_list)

    for top_k in tqdm(top_k_list):

        # get matches and hops
        df = calculate_matches(df, top_k)

        # get results for all hops 
        for num_hops in num_hops_list:


            if num_hops == 'all':
                df_hop = df[['number_of_matches', 'num_hops']]
            else:
                df_hop = df[df['num_hops'].isin([num_hops])][['number_of_matches', 'num_hops']]
        

            hit = (df_hop['number_of_matches']==df_hop['num_hops']).sum() / len(df_hop)
            df_hits.loc[num_hops, top_k] = hit

            recall = df_hop.sum()['number_of_matches']/df_hop.sum()['num_hops']
            df_recall.loc[num_hops, top_k] = recall
            
            
    return df_hits, df_recall


def extract_top_docs(df_DB, df, top_k):
    
    df_DB['fact']=df_DB['article_name']+ '. ' + df_DB['text']
    map_title_fact = {index: fact for index, fact in df_DB['fact'].items()}
    df_top_docs = pd.DataFrame(top_k, columns = np.arange(top_k.shape[1])+1)
    df_top_docs = df_top_docs.add_prefix('retrieved_doc_')
    df_top_docs = df_top_docs.applymap(map_title_fact.get)
    df = pd.concat([df, df_top_docs], axis=1)
    
    #df.drop(columns=['uid', 'label', 'hpqa_id'], inplace = True)
    return df


def map_documents(df_DB, df, mips_top_indices):
    """
    Maps the top_k retrieved documents from the wiki database to the original dataframe.

    Args:
        df_DB (pandas.DataFrame): Dataframe containing the database documents.
            It should have columns 'article_name' and 'text'.
        df (pandas.DataFrame): hover dataframe to which the retrieved documents will be mapped.
        top_k (numpy.ndarray): 2D array representing the retrieved document indices for each claim.

    Returns:
        pandas.DataFrame: Updated dataframe with retrieved documents mapped.

    """
    # Combine 'article_name' and 'text' columns in the database dataframe to form 'fact' column
    df_DB['fact'] = df_DB['article_name'] + SEP_TOKEN + df_DB['text']

    # Create a mapping of index to fact in the database dataframe
    map_title_fact = {index: fact for index, fact in df_DB['fact'].items()}

    # Create a dataframe for retrieved documents with column names 'retrieved_doc_1', 'retrieved_doc_2', etc.
    df_top_docs = pd.DataFrame(mips_top_indices, columns=np.arange(mips_top_indices.shape[1]) + 1)
    df_top_docs = df_top_docs.add_prefix('retrieved_doc_')

    # Map the retrieved documents using the index-to-fact mapping
    df_top_docs = df_top_docs.applymap(map_title_fact.get)

    # Concatenate the original dataframe with the retrieved document dataframe
    df = pd.concat([df, df_top_docs], axis=1)

    return df


def get_rank_supporting_facts(df, mips_top_indices):
    """
    This function takes a DataFrame and a list of top indices and ranks the supporting facts.
    Each row of the DataFrame corresponds to a query. The 'supporting_facts_ids' column 
    contains the IDs of the documents that support the query. The function compares these 
    IDs with the top indices returned by an information retrieval system (e.g., MIPS).

    Args:
    df (pd.DataFrame): The DataFrame that includes a 'supporting_facts_ids' column. Each 
                       element in this column is a tensor of IDs for supporting documents.
    mips_top_indices (list): A list of tensors. Each tensor contains the indices of the 
                             top documents returned by the information retrieval system. 
                             Each element in the list corresponds to a query in the DataFrame.

    Returns:
    df_rank (pd.DataFrame): A new DataFrame where each column corresponds to a supporting 
                            fact and the values are the ranks of these facts. If a supporting 
                            fact is not in the top indices, its rank is -1.
    """

    # Initialize a DataFrame to store the ranks
    df_rank = pd.DataFrame(columns=[1,2,3,4])
    df_rank = df_rank.astype('float32')

    # Loop over the rows in the original DataFrame
    for row in tqdm(range(df.shape[0])):    

        # Get the IDs of the supporting documents for the current query
        tensor_relevant_docs = torch.tensor(df.loc[row,'supporting_facts_ids'])
        
        # Get the top indices for the current query
        tensor_top_k = mips_top_indices[row]

        # Loop over the supporting documents
        for i, value in enumerate(tensor_relevant_docs):

            # Find where the current document appears in the top indices
            indices = torch.where(tensor_top_k == value)

            # If the document is in the top indices, store its rank (index + 1)
            if indices[0].numel() > 0:
                df_rank.loc[row,i+1] = indices[0].item() + 1
            # If the document is not in the top indices, its rank is -1
            else:
                df_rank.loc[row,i+1] = -1

    # Add a prefix to the column names for clarity
    df_rank = df_rank.add_prefix('supporting_fact_')
    
    
    return df_rank


# Custom formatter function
def to_percentage(x, pos):
    return f'{x * 100:.0f}%'

# plotting functions

def read_results_all(metric, models, label, legend_names):
    
    df_plot_hops = pd.DataFrame()
    
    # Create dictionary of DataFrames
    df_dict = {
        model: pd.read_csv(
            f'/pfs/data5/home/kit/stud/ulhni/nlp_project/embeddings/results/df_{metric}_{model}_{label}.csv', 
            index_col=0
        ).T
        for model in models
    }

    # For df_plot DataFrame
    df_plot = pd.concat(
        [df_dict[model].iloc[:, -1] for model in models], 
        axis=1
    )
    
    # Assign new column names
    df_plot.columns = legend_names

    # For df_plot_hops DataFrame
    if 'tfidf' in df_dict.keys():
        if 'bs128' in df_dict.keys():
            df_plot_hops = pd.concat(
                [
                    df_dict['tfidf'].iloc[:, :3].add_suffix('_hops').add_prefix('TF-IDF_'), 
                    df_dict['bs128'].iloc[:, :3].add_suffix('_hops').add_prefix('BERT_128_')
                ], 
                axis=1
            )
    elif 'bs128' in df_dict.keys():
        df_plot_hops =  df_dict['bs128'].iloc[:, :3].add_suffix('_hops').add_prefix('BERT_128_')
 
    
    
    return df_dict, df_plot, df_plot_hops


def plot_all(ax, df_plot, colors, metric):  
    # Plotting for the first subplot
    for j, col in enumerate(df_plot.columns):
        if col.startswith('TF-IDF'): 
            ax.plot(df_plot[col], linewidth=3, linestyle='dotted', marker = 'o', markersize=9, color=colors[j]) 
        else:
            ax.plot(df_plot[col], linewidth=3,  marker = 'o', markersize=9, color=colors[j])

    #ax.legend(df_plot.columns, fontsize = 13, loc='upper left')
    ax.set_ylabel(f'{metric}', fontsize=20, labelpad=20)
    ax.set_xlabel('k: # of retrieved documents', fontsize=20, labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(to_percentage))

    return ax

def plot_hops(ax, df_plot_hops, colors_hops, metric):  

    # Plotting for the second subplot
    for j, col in enumerate(df_plot_hops.columns):
        if col.startswith('TF-IDF'): 
            ax.plot(df_plot_hops[col], linewidth=3, linestyle='dotted', marker = 'o', markersize=9, color=colors_hops[int(col[-6])]) 
        else:
            ax.plot(df_plot_hops[col], linewidth=3,  marker = 'o', markersize=9, color=colors_hops[int(col[-6])])

    #ax.legend(df_plot_hops.columns, fontsize = 13, loc='upper left')
    ax.set_ylabel(f'{metric}', fontsize=20, labelpad=20)
    ax.set_xlabel('k: # of retrieved documents', fontsize=20, labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(to_percentage))

    return ax


def read_results_label(metric, models, names):
    # create the df for the supported and non supported claims

    # Define the support statuses
    support_statuses = ['SUPPORTED', 'NOT_SUPPORTED']

    # Create dictionaries of DataFrames for each support status
    df_dicts = {}
    for support_status in support_statuses:
        df_dict = {
            model: pd.read_csv(
                f'/pfs/data5/home/kit/stud/ulhni/nlp_project/embeddings/df_{metric}_{model}_{support_status}.csv',
                index_col=0
            ).T
            for model in models
        }
        df_dicts[support_status] = df_dict

    # For df_plot DataFrame
    df_plot = pd.concat(
        [df_dicts[support_status][model].iloc[:, -1] for model in models for support_status in support_statuses],
        axis=1
    )

    # Assign new column names
    df_plot.columns = names

    # For df_plot_hops DataFrame
    if 'tfidf' in df_dict.keys():
        df_plot_hops = pd.concat(
            [
                pd.concat([df_dicts[support_status]['tfidf'].iloc[:, :3].add_suffix(f'_hops').add_prefix(f'TF-IDF_{support_status}_') for support_status in support_statuses], axis=1),
                pd.concat([df_dicts[support_status]['bs128'].iloc[:, :3].add_suffix(f'_hops').add_prefix(f'BERT_128_{support_status}_') for support_status in support_statuses], axis=1)
            ],
            axis=1
        )
    else:
        df_plot_hops = pd.concat(
        [
            df_dicts[support_status]['bs128'].iloc[:, :3].add_suffix(f'_hops').add_prefix(f'BERT_128_{support_status}_') 
            for support_status in support_statuses
        ], 
        axis=1
    )
        
    return df_plot, df_plot_hops


# plot of supported and not-supported docs
def plot_on_label(ax, df_plot, metric, colors):
    for j, col in enumerate(df_plot.columns):

        if 'NOT_SUP' in col:
            c = colors[0]
        else:
            c = colors[1]

        if col.startswith('TF-IDF'): 
            ax.plot(df_plot[col], linewidth=3, linestyle='dotted', marker = 'o', markersize=9, color=c) 
        else:
            ax.plot(df_plot[col], linewidth=3,  marker = 'o', markersize=9, color=c)


    ax.set_ylabel(f'{metric}', fontsize=20, labelpad=20)
    ax.set_xlabel('k: # of retrieved documents', fontsize=20, labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(to_percentage))

    return ax


def add_relevancy_info(df, top_k):
    for row in tqdm(range(len(df))):
        for i, idx in enumerate(df.loc[row, 'retrieval_ids'][:top_k]):
            j = i+1
            if idx in df.loc[row, 'supporting_facts_ids']:
                df.loc[row, f'retrieved_doc_{j}'] = "T - " + df.loc[row, f'retrieved_doc_{j}']
            else:
                df.loc[row, f'retrieved_doc_{j}'] = "F - " + df.loc[row, f'retrieved_doc_{j}']

    return df


def shorten_text(text, max_len=200):
        return (text[:max_len] + '...') if len(text) > max_len else text

def shorten_text_claim(text, max_len=300):
    return (text[:max_len] + '...') if len(text) > max_len else text



def create_hover_text(df_hover, df_tfidf, max_len):
    
    # Define a lambda function to apply shorten_text with the given max_len
    shorten = lambda text: shorten_text(text, max_len=max_len)

    
    hover_text = " Claim: " + df_hover['claim'].apply(shorten_text_claim) + \
                 "<br>" + \
                 "<br>Metrics:" + \
                 "<br>" + \
                 "<br>Recall_bert: " + df_hover['recall'].astype(str) + \
                 "<br>Hits_bert: " + df_hover['hits'].astype(str) + \
                 "<br>Recall_tfidf: " + df_tfidf['recall'].astype(str) + \
                 "<br>Hits_tfidf: " + df_tfidf['hits'].astype(str) + \
                 "<br>" + \
                 "<br>Retrieved Docs:" + \
                 "<br>"

    doc_numbers = list(range(1, 6))
    for num in doc_numbers:
        hover_text += "<br>BERT doc " + str(num) + ": " + df_hover['retrieved_doc_'+str(num)].apply(shorten)
    hover_text +=  "<br>"
    for num in doc_numbers:
        hover_text += "<br>TFIDF doc " + str(num) + ": " + df_tfidf['retrieved_doc_'+str(num)].apply(shorten)

    
    return hover_text


def figure_layout(fig, width=1600, height=800):

    fig.update_layout(
        
        
        
        legend=dict(
            x=1.1,
            y=0.48,
            font=dict(  # Update the font size of the legend
                size=18  # Set font size to 14
            )
        ),
        xaxis=dict(
            title='Dimension 1',
            gridcolor='lightgrey',  # Set the color of the grid lines to light gray
            gridwidth=0.5,  # Set the width of the grid lines
            showline=True,  # Show the line for the x-axis
            linecolor='lightgrey',  # Set the color of the x-axis line to black
            linewidth=2,  # Set the width of the x-axis line
            zeroline=False,
            tickfont=dict(  # Update the font size of the xticks
                size=16  # Set font size to 14
            ),
            titlefont=dict(  # Update the font size of the xaxis title
                size=18  # Set font size to 16
            )
        ),
        yaxis=dict(
            title='Dimension 2',
            gridcolor='lightgrey',  # Set the color of the grid lines to light gray
            gridwidth=0.5,  # Set the width of the grid lines
            showline=True,  # Show the line for the y-axis
            linecolor='lightgrey',  # Set the color of the y-axis line to black
            linewidth=2,# Set the width of the y-axis line
            zeroline=False,
            tickfont=dict(  # Update the font size of the yticks
                size=16  # Set font size to 14
            ),
            titlefont=dict(  # Update the font size of the yaxis title
                size=18  # Set font size to 16
            )
        ),
        autosize=False,
        plot_bgcolor='white',  # Set the background color to white
        width=width,  # Width in pixels
        height=height,  # Height in pixels
    )

    return fig


# function for creating latex tables
def results_latex(df):
    df = df.round(decimals=2).T
    df['mean'] = df.mean(axis=1)
    df.columns = pd.MultiIndex.from_product([['k'], df.columns])
    results = df.to_latex(float_format="%.2f")
    results = results.replace("[t]", "")
    results = results.replace("BERT_", "BERT\_")

    print(results)