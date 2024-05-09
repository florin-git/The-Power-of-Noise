#!/bin/bash

# cd ..

# Leave --percentages_for_index_splitting empty to create only one index with all embeddings
# --percentages_for_index_splitting 60 means that two indices will be created, one with 60% of the embeddings and another with 40%

python src/index_embeddings.py \
    --corpus_size 21035236 \
    --vector_sz 768 \
    --idx_type IP \
    --faiss_dir data/corpus/faiss/wiki_dec_2018/to_gpu \
    --percentages_for_index_splitting 60 \
    --output_dir data/corpus/embeddings/wiki_dec_2018 \
    --prefix_name contriever \
    --batch_size 512 \
    --save_every 500