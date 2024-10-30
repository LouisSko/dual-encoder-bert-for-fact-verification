#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu_8
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=ft_bert_test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ulhni@student.kit.edu
#SBATCH --output=/home/kit/stud/ulhni/nlp_project/training_results/bs16_no_dropout_pooled_output/output.txt
#SBATCH --error=/home/kit/stud/ulhni/nlp_project/training_results/bs16_no_dropout_pooled_output/error.txt

# Activate env
source /home/kit/stud/ulhni/nlp_project/nlp_env/bin/activate

# Run Python script
python /home/kit/stud/ulhni/nlp_project/finetuning_bert_test.py

# make executable: chmod +x /home/kit/stud/ulhni/nlp_project/job_train_test.sh
# run: sbatch /home/kit/stud/ulhni/nlp_project/job_train_test.sh