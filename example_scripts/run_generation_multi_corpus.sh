#!/bin/bash

# cd ..

# llm_id microsoft/phi-2
# llm_id tiiuae/falcon-7b-instruct
# llm_id mosaicml/mpt-7b-instruct

CUDA_VISIBLE_DEVICES=0 python src/generate_answers_llm_multi_corpus.py \
    --output_dir data/gen_res \
    --llm_id mosaicml/mpt-7b-instruct \
    --model_max_length 2048 \
    --load_full_corpus False \
    --use_bm25 False \
    --use_corpus_nonsense False \
    --num_main_documents 1 \
    --num_other_documents 1 \
    --put_main_first False \
    --use_test True \
    --batch_size 18 \
    --save_every 250

