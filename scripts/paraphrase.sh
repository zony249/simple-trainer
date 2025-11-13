#!/bin/bash 
#SBATCH --account=aip-lilimou 
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8 
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=64G
#SBATCH --time=1-00:00
#SBATCH --job-name=compression-llama7b
#SBATCH --output=logs/%j--paraphrase.out


nvidia-smi 
nvidia-smi topo -m 

export CUDA_VISIBLE_DEVICES=4,5
# export HF_HUB_OFFLINE=1

python3 -m simple-trainer.llm-paraphrase \
    --hf_name_or_path=Qwen/Qwen3-8B \
    --task=alpaca_plus \
    --save_dir=runs \