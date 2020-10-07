#!/bin/bash
#SBATCH --output=/scratch/pmh330/lm_evaluation_harness/logs/tense-%j.out
#SBATCH --error=/scratch/pmh330/lm_evaluation_harness/logs/tense-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --job-name=gpt

# Preprocess, download, tokenization
# must point to transformers repo currently
#conda activate /scratch/pmh330/virtual_envs/v_jiant2
export OPENAI_API_SECRET_KEY="sk-ByOtCfCE88G6xH9APoXODFL1tw5Ed4Z6i005IuA9" 
conda activate /scratch/pmh330/virtual_envs/v_gpt

/scratch/pmh330/virtual_envs/v_gpt/bin/python main.py --model gpt3 --tasks $1 --provide_description --num_fewshot $2 --run_mc_validation
