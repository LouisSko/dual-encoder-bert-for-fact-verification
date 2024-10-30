#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu_8
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:8
#SBATCH --job-name=finetuning_bert
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ulhni@student.kit.edu
#SBATCH --output=/home/kit/stud/ulhni/nlp_project/training_results/roberta_bs200/output.txt
#SBATCH --error=/home/kit/stud/ulhni/nlp_project/training_results/roberta_bs200/error.txt

# Activate env
source /home/kit/stud/ulhni/nlp_project/nlp_env/bin/activate

# Run Python script
python /home/kit/stud/ulhni/nlp_project/finetuning_roberta.py

# make executable: chmod +x /home/kit/stud/ulhni/nlp_project/job_train_roberta.sh
# run: sbatch /home/kit/stud/ulhni/nlp_project/job_train_roberta.sh