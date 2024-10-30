#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu_8
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=10000mb
#SBATCH --job-name=encoding
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ulhni@student.kit.edu
#SBATCH --output=/home/kit/stud/ulhni/nlp_project/embeddings/bs128_unit_length/output.txt
#SBATCH --error=/home/kit/stud/ulhni/nlp_project/embeddings/bs128_unit_length/error.txt

# Activate env
source /home/kit/stud/ulhni/nlp_project/nlp_env/bin/activate

# Run Python script
python /home/kit/stud/ulhni/nlp_project/encode_claims_docs.py

# make executable: chmod +x /home/kit/stud/ulhni/nlp_project/job_encode.sh
# run: sbatch /home/kit/stud/ulhni/nlp_project/job_encode.sh