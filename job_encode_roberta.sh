#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu_8
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4
#SBATCH --job-name=encoding
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ulhni@student.kit.edu
#SBATCH --output=/home/kit/stud/ulhni/nlp_project/embeddings/roberta_bs128/output.txt
#SBATCH --error=/home/kit/stud/ulhni/nlp_project/embeddings/roberta_bs128/error.txt

# Activate env
source /home/kit/stud/ulhni/nlp_project/nlp_env/bin/activate

# Run Python script
python /home/kit/stud/ulhni/nlp_project/encode_claims_docs_roberta.py

# make executable: chmod +x /home/kit/stud/ulhni/nlp_project/job_encode_roberta.sh
# run: sbatch /home/kit/stud/ulhni/nlp_project/job_encode_roberta.sh