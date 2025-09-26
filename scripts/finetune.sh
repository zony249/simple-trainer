#!/bin/bash 

export CUDA_VISIBLE_DEVICES=1,2,3
# export DEBUGPY_ENABLE=1


python -m debugpy --listen 0.0.0.0:5678 -m \
    accelerate.commands.launch \
    --config_file accel_config/fsdp2.yaml \
    -m simple-trainer \
        --hf_name_or_path=Qwen/Qwen3-0.6B \
        --task=alpaca_plus \
        --lr=5e-5 \
        --epochs=3 \
        --batch_size=2 \
        --gradient_accumulation_steps=256 \