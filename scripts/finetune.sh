#!/bin/bash 

export CUDA_VISIBLE_DEVICES=4,5,6
export DEBUGPY_ENABLE=1


accelerate launch \
    --num_processes 3 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --main_process_port 12345 \
    --config_file accel_config/fsdp.yaml \
    -m simple-trainer \
        --hf_name_or_path=huggyllama/Llama-7b \
        --task=wikitext \
        --lora_adapter=random_init \
        --lr=5e-5 \
        --epochs=3 \
        --batch_size=2 \
        --gradient_accumulation_steps=4 \