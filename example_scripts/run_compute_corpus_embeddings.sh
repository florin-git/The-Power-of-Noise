#!/bin/bash

# cd ..

python src/compute_corpus_embeddings.py \
    --corpus_path data/corpus/wiki_dec_2018.json \
    --encoder_id facebook/contriever \
    --max_length_encoder 512 \
    --output_dir data/corpus/embeddings/wiki_dec_2018 \
    --prefix_name contriever \
    --batch_size 512 \
    --save_every 500 \

