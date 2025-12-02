#!/bin/bash 
#SBATCH --account=aip-lilimou 
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8 
#SBATCH --gpus-per-node=l40s:4
#SBATCH --mem=400G
#SBATCH --time=0-23:00
#SBATCH --job-name=mi-comp-llama7b
#SBATCH --output=logs/%j--mi-comp-llama7b--gist.out

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1
export HF_HOME=$SCRATCH
export TAG="mi-gist"
export DEBUGPY_ENABLE=0

export NCCL_P2P_DISABLE=1

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO


nvidia-smi 
nvidia-smi topo -m

mkdir -p $SCRATCH/runs

accelerate launch \
    --config_file accel_config/ddp.yaml \
    -m simple-trainer \
        --hf_name_or_path=llama-7b \
        --task=alpaca_pp \
        --lr=2e-4 \
        --epochs=4 \
        --batch_size=2 \
        --gradient_accumulation_steps=10 \
        --eval_steps=2000 \
        --output_dir=$SCRATCH/runs/$(date +%Y-%m-%d--%H-%M-%S)--$TAG \
        --lora_adapter=random_init \
        --alpha=1e-3 \