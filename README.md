# NLP Project: Information Retrieval with Dual Encoder Models

This repository contains resources for fine-tuning Dual Encoder BERT and RoBERTa models, encoding documents and claims, demonstrating fact retrieval, and computing results for an NLP project focused on information retrieval. The project utilises the principle of in-batch negatives and TF-IDF retrieved documents for the fine-tuning process. 

**Note:** Due to the large size of certain data files, they are not included in the repository.

## Repository Structure

### Fine-tuning Scripts

- **finetuning_bert.py:** Fine-tunes a Dual Encoder BERT model based on in-batch negatives.
- **finetuning_roberta.py:** Fine-tunes a Dual Encoder RoBERTa model based on in-batch negatives.
- **finetuning_bert_tfidf.py:** Fine-tunes a Dual Encoder BERT model based on a dataset using the top retrieved documents from TF-IDF.

### Document and Claims Encoding Scripts

- **encode_claims_docs.py:** Handles the encoding of claims and documents for Dual Encoder BERT models.
- **encode_claims_docs_roberta.py:** Handles the encoding of claims and documents for Dual Encoder RoBERTa models.

### Demonstration Scripts

The scripts **demo.ipynp** and **demo.py** provide a demonstration of the fact retrieval process. However, they are not executable in their current state as the embeddings of the wiki database (exceeding 20GB in size) are not provided in this repository.

### Results Calculation Script

- The scripts **nlp_results.ipynb** and **nlp_results.py** computes and presents results and plots based on the models and encoding processes described above.

### HPC Cluster Scripts

Files starting with "job_" are bash scripts used for submitting jobs to a High-Performance Computing (HPC) cluster.

### Directories Lacking Files

The directories "embeddings" and "hover-main" do not contain any files due to the large data sizes associated with these files.
