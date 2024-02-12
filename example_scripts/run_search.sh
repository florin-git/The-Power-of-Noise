#!/bin/bash

# cd ..

python src/compute_search_results.py \
    --faiss_dir data/faiss/ \
    --use_index_on_gpu True  \
    --gpu_ids 0 3 \
    --use_test True

