##!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
export TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
export TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
YOUR_LLM_PATH="meta-llama/Llama-2-7b-hf"
YOUR_UNS_DATA_PATH="//workspace//INFO-RAG//result"
YOUR_DATA_SAVE_PATH="//workspace//INFO-RAG//info_rag_result"
ZERO_STAGE=3

#mkdir -p $OUTPUT

/usr/local/bin/deepspeed --num_gpus=1 --master_addr="127.0.0.1" --master_port=29560 train_info_rag.py \
   --model_name_or_path $YOUR_LLM_PATH \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --setting_max_length 1024 \
   --uns_data_path $YOUR_UNS_DATA_PATH \
   --learning_rate 1e-5 \
   --weight_decay 0. \
   --num_train_epochs 2  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --lora_dim 128 \
   --lora_module_name "layers." \
   --output_dir $YOUR_DATA_SAVE_PATH
