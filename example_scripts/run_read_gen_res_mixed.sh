#!/bin/bash

# cd ..

# llm_id microsoft/phi-2
# llm_id tiiuae/falcon-7b-instruct
# llm_id mosaicml/mpt-7b-instruct
# llm_id meta-llama/Llama-2-7b-chat-hf

python src/read_generation_results.py \
    --output_dir data/gen_res \
    --llm_id microsoft/phi-2  \
    --use_test True \
    --prompt_type mixed \
    --use_bm25 False \
    --num_retrieved_documents 1 \
    --num_random_documents 1 \
    --put_retrieved_first False \

