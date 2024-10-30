#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu_8
#SBATCH --time=07:00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=10000mb
#SBATCH --job-name=train_bert
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ulhni@student.kit.edu
#SBATCH --output=/home/kit/stud/ulhni/nlp_project/training_results/bs16_tfidf/output.txt
#SBATCH --error=/home/kit/stud/ulhni/nlp_project/training_results/bs16_tfidf/error.txt

# Activate env
source /home/kit/stud/ulhni/nlp_project/nlp_env/bin/activate

# Run Python script
python /home/kit/stud/ulhni/nlp_project/finetuning_bert_tfidf.py

# make executable: chmod +x /home/kit/stud/ulhni/nlp_project/job_train_bert_tfidf.sh
# run: sbatch /home/kit/stud/ulhni/nlp_project/job_train_bert_tfidf.sh