#!/bin/bash

# cd ..

# llm_id microsoft/phi-2
# llm_id tiiuae/falcon-7b-instruct
# llm_id mosaicml/mpt-7b-instruct
# llm_id meta-llama/Llama-2-7b-chat-hf

python src/read_generation_results.py \
    --output_dir data/gen_res \
    --llm_id mosaicml/mpt-7b-instruct \
    --use_test True \
    --prompt_type multi_corpus \
    --use_bm25 False \
    --use_corpus_nonsense False \
    --num_main_documents 1 \
    --num_other_documents 1 \
    --put_main_first False \

