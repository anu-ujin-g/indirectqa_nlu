#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=v100_sxm2_4,p40_4,p100_4,k80_4
#SBATCH --mem=64000
#SBATCH --time=48:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ago265@nyu.edu
#SBATCH --job-name="testing"
#SBATCH --output=/scratch/ago265/nlu_project/outputs/%j.out

module purge
module load anaconda3/2020.07
module load cuda/11.1.74
module load gcc/10.2.0

# Replace with your NetID
NETID=ago265
source activate nmt_env

# Set project working directory
PROJECT=/scratch/${NETID}/nlu_project

# Set arguments
STUDY_NAME=t5strict_baseline
SAVE_DIR=${PROJECT}/saved_models
DATA_DIR=${PROJECT}/indirectqa_nlu/Data
EPOCHS=3


cd ${PROJECT}
python ./indirectqa_nlu/scripts/train_attention.py \
	--experiment ${STUDY_NAME} \
	--save_dir ${SAVE_DIR} \
	--data_dir ${DATA_DIR} \
	--epochs ${EPOCHS} \
