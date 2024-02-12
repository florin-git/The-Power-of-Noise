#!/bin/bash

# cd ..

# llm_id microsoft/phi-2
# llm_id tiiuae/falcon-7b-instruct
# llm_id mosaicml/mpt-7b-instruct

CUDA_VISIBLE_DEVICES=0 python src/generate_answers_llm_only_query.py \
    --output_dir data/gen_res \
    --llm_id meta-llama/Llama-2-7b-chat-hf \
    --model_max_length 4096 \
    --use_test False \
    --batch_size 12 \
    --save_every 250

