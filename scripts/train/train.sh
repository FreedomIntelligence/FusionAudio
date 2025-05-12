#!/bin/bash
export TRANSFORMERS_CACHE=/wangbenyou/xinyuan/hf_cache/
export HF_DATASETS_CACHE=/wangbenyou/xinyuan/hf_cache/

#export CUDA_VISIBLE_DEVICES=4,5,6,7

output_dir='/wangbenyou/xinyuan/models/'

mkdir -p $output_dir
cp "$0" ${output_dir}/$(date +"%Y-%m-%d-%H-%M-%S").sh

torchrun --nproc_per_node=8 --master_port=1234 /wangbenyou/xinyuan/ECHO-AQA/src/GAMA/finetune.py \
    --base_model '/wangbenyou/xinyuan/models/GAMA4-2/pytorch_model.bin' \
    --data_path '/wangbenyou/xinyuan/ECHO-AQA/data/ECHO-AQA/ECHO-high-25k.json' \
    --output_dir $output_dir \
    --batch_size 256 \
    --micro_batch_size 4 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --cutoff_len 256 \
    --val_set_size 200 \
    --warmup_steps 9 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --wandb_run_name ${output_dir} \
    --save_steps 50 \
    --eval_steps 30 \
    --trainable_params qformer_all
