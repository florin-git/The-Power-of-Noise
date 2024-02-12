#!/bin/bash

# cd ..

# llm_id microsoft/phi-2
# llm_id tiiuae/falcon-7b-instruct
# llm_id mosaicml/mpt-7b-instruct

CUDA_VISIBLE_DEVICES=0 python src/generate_answers_llm_mixed.py \
    --output_dir data/gen_res \
    --llm_id microsoft/phi-2 \
    --model_max_length 2048 \
    --load_full_corpus False \
    --use_bm25 False \
    --num_retrieved_documents 1 \
    --num_random_documents 2 \
    --put_retrieved_first False \
    --use_test True \
    --batch_size 16 \
    --save_every 250

