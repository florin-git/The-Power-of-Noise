#!/bin/bash

# cd ..

python src/compute_search_results.py \
    --faiss_dir data/corpus/faiss/wiki_dec_2018/to_gpu \
    --use_index_on_gpu True  \
    --gpu_ids 0 1 \
    --vector_sz 768 \
    --encoder_id facebook/contriever \
    --max_length_encoder 512 \
    --top_docs 150 \
    --use_test True \
    --output_dir data/search_results \
    --prefix_name contriever \
    --batch_size 512 \
    --index_batch_size 8
