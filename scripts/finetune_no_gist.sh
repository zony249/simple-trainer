#!/bin/bash 
#SBATCH --account=aip-lilimou 
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8 
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=400G
#SBATCH --time=3-00:00
#SBATCH --job-name=llama7b
#SBATCH --output=logs/llama7b--no-gist-%j.out

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1
export HF_HOME=$SCRATCH
export TAG="no-gist"
# export DEBUGPY_ENABLE=1
nvidia-smi 
nvidia-smi topo -m

# python -m debugpy --listen 0.0.0.0:5678 -m \
    # accelerate.commands.launch \
    # --config_file accel_config/fsdp2.yaml \
# --config_file accel_config/fsdp2.yaml \
accelerate launch \
    --config_file accel_config/fsdp2.yaml \
    -m simple-trainer \
        --hf_name_or_path=llama-7b \
        --task=alpaca_plus \
        --lr=1e-5 \
        --epochs=4 \
        --batch_size=4 \
        --gradient_accumulation_steps=32 \
        --eval_steps=2000 \
        --output_dir=runs/$(date +%Y-%m-%d--%H-%M-%S)--$TAG \
        --num_gist_tokens=0 \