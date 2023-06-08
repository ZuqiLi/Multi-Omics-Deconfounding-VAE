#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=14G
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o log_cluster/out_%j.log
#SBATCH -e log_cluster/error_%j.log

#Load the modules & venv
module purge
#module load Python/3.9.5-GCCcore-10.3.0
source "/tmp/${SLURM_JOB_USER}.${SLURM_JOB_ID}/prolog.env"  # this is in tutorial script -- what does it do?

# ACTIVATE ANACONDAi
eval "$(conda shell.bash hook)"
source activate env_multiviewVAE
echo $CONDA_DEFAULT_ENV


python scripts/trainModel_adversarial_multiCov_1batch.py
