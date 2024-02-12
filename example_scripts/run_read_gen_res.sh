#!/bin/bash

# cd ..

python src/read_generation_results.py \
    --output_dir data/gen_res \
    --llm_id meta-llama/Llama-2-7b-chat-hf \
    --use_test False \
    --prompt_type classic \
    --use_random True \
    --use_adore False \
    --gold_position 0 \
    --num_documents_in_context 1 \
    --get_documents_without_answer True \

