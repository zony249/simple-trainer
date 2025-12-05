#!/bin/bash 
#SBATCH --account=aip-lilimou 
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8 
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=400G
#SBATCH --time=3-00:00
#SBATCH --job-name=compression-llama7b
#SBATCH --output=logs/%j--compression-llama7b--gist--rerun.out

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=0
export HF_HOME=$SCRATCH
export TAG="debug"
export DEBUGPY_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

nvidia-smi 
nvidia-smi topo -m

# python -m debugpy --listen 0.0.0.0:5678 -m \
    # accelerate.commands.launch \
    # --config_file accel_config/fsdp2.yaml \
# --config_file accel_config/fsdp2.yaml \

accelerate launch \
    --config_file accel_config/ddp.yaml \
    -m simple-trainer \
        --hf_name_or_path=llama-7b \
        --task=alpaca_pp \
        --lr=2e-4 \
        --epochs=1 \
        --batch_size=1 \
        --gradient_accumulation_steps=32 \
        --eval_steps=2000 \
        --lora_adapter="random_init" \
        --output_dir=runs/$(date +%Y-%m-%d--%H-%M-%S)--$TAG \
        --alpha=1e-2 \

# accelerate launch \
#     --config_file accel_config/fsdp2.yaml \
#     -m simple-trainer.models.cllama

# python -m simple-trainer.models.cllama