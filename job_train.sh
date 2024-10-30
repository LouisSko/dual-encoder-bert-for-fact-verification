#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu_8
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:8
#SBATCH --job-name=finetuning_bert
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ulhni@student.kit.edu
#SBATCH --output=/home/kit/stud/ulhni/nlp_project/training_results/bs200_no_dropout/output.txt
#SBATCH --error=/home/kit/stud/ulhni/nlp_project/training_results/bs200_no_dropout/error.txt

# Activate env
source /home/kit/stud/ulhni/nlp_project/nlp_env/bin/activate

# Run Python script
python /home/kit/stud/ulhni/nlp_project/finetuning_bert.py

# make executable: chmod +x /home/kit/stud/ulhni/nlp_project/job_train.sh
# run: sbatch /home/kit/stud/ulhni/nlp_project/job_train.sh