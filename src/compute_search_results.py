import os 
import pickle
import argparse
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoConfig

from index import Indexer, merge_ip_search_results
from retriever import Encoder, Retriever
from utils import *


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED=10

info = {
    "train": {
        "data_path": 'data/10k_train_dataset.json',
    },
    "test": {
        "data_path": 'data/test_dataset.json',
    }
}

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search and Index Utility.")
    parser.add_argument('--faiss_dir', type=str, help='Directory containing FAISS index data')
    parser.add_argument('--use_index_on_gpu', type=str2bool, help='Flag to use index on GPU')
    parser.add_argument('--gpu_ids', nargs='+', type=int, help='GPU IDs for indexing, required if --use_index_on_gpu is set')
    parser.add_argument('--vector_sz', type=int, default=768, help='Size of the vectors to be indexed')
    parser.add_argument('--idx_type', type=str, default='IP', help='Index type (IP for Inner Product)')
    parser.add_argument('--encoder_id', type=str, default='facebook/contriever', help='Model identifier for the encoder')
    parser.add_argument('--max_length_encoder', type=int, default=512, help='Maximum sequence length for the encoder')
    parser.add_argument('--normalize_embeddings', type=str2bool, default=False, help='Whether to normalize embeddings')
    parser.add_argument('--lower_case', type=str2bool, default=False, help='Whether to lower case the corpus text')
    parser.add_argument('--do_normalize_text', type=str2bool, default=True, help='Whether to normalize the corpus text')
    parser.add_argument('--top_docs', type=int, default=150, help='Number of documents to retriever from similarity search')
    parser.add_argument('--use_test', type=str2bool, help='Use the test set', default=False)
    parser.add_argument('--output_dir', type=str, default='data/faiss', help='Output directory for saving search results')
    parser.add_argument('--prefix_name', type=str, default='contriever', help='Initial part of the name of the saved search results')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for encoding queries')
    parser.add_argument('--index_batch_size', type=int, default=64, help='Batch size for FAISS search operations')
    
    args =  parser.parse_args()
    args.split = "test" if args.use_test else "train"

    # Validate gpu_ids if use_index_on_gpu is True
    if args.use_index_on_gpu and (args.gpu_ids is None or len(args.gpu_ids) == 0):
        parser.error('--gpu_ids must be set when --use_index_on_gpu is used.')

    return args


def load_queries(args: argparse.Namespace) -> List[str]:
    """Load queries from dataset."""

    df = pd.read_json(info[args.split]['data_path'])
    queries = df['query'].tolist() if 'query' in df.columns else df['question'].tolist()
    return queries


def initialize_index(args: argparse.Namespace) -> List[Indexer]:
    """Initialize and deserialize FAISS indexes."""
    indexes = []
    if args.use_index_on_gpu:
        for i, gpu_id in enumerate(args.gpu_ids):
            index = Indexer(args.vector_sz, idx_type=args.idx_type)
            index.deserialize_from(
                args.faiss_dir, 
                f'IP_index{i+1}.faiss', f'IP_index{i+1}_meta.faiss', 
                gpu_id=gpu_id
            )
            indexes.append(index)
    else: # CPU
        index = Indexer(args.vector_sz, idx_type=args.idx_type)
        index.deserialize_from(args.faiss_dir)
        indexes.append(index)
    return indexes


def initialize_retriever(args: argparse.Namespace) -> Retriever:
    """Initialize the encoder and retriever."""
    config = AutoConfig.from_pretrained(args.encoder_id)
    encoder = Encoder(config).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_id)
    retriever = Retriever(
        device=device, tokenizer=tokenizer, 
        query_encoder=encoder, 
        max_length=args.max_length_encoder,
        norm_query_emb=args.normalize_embeddings,
        lower_case=args.lower_case,
        do_normalize_text=args.do_normalize_text
    )

    return retriever


def process_queries(retriever: Retriever, queries: List[str], batch_size: int) -> np.ndarray:
    """Encode queries using the retriever."""
    return retriever.encode_queries(queries, batch_size=batch_size).numpy()


def search_documents(
    args: argparse.Namespace,
    indexes: List[Indexer], 
    query_embeddings: np.ndarray, 
) -> List[Tuple[List[str], List[float]]]:
    """Search documents using the indexes."""
    if args.use_index_on_gpu:
        search_results = merge_ip_search_results(
            indexes[0], indexes[1], query_embeddings, 
            top_docs=args.top_docs, 
            index_batch_size=args.index_batch_size
        )
    else:
        search_results = indexes[0].search_knn(
            query_embeddings, top_docs=args.top_docs, 
            index_batch_size=args.index_batch_size
        )
    return search_results


def save_search_results(
    args: argparse.Namespace,
    search_results: List[Tuple[List[str], List[float]]], 
):
    """Save search results to a pickle file."""
    os.makedirs(args.output_dir, exist_ok=True)
    file_path = os.path.join(
        args.output_dir, f'{args.prefix_name}_{args.idx_type}_{args.split}_search_results_at{args.top_docs}.pkl'
    )
    write_pickle(search_results, file_path)



def main():
    args = parse_arguments()

    print("Loading data...")
    queries = load_queries(args)
    print("Data loaded")

    print("Building index...")
    indexes = initialize_index(args)
    print("Index loaded")

    retriever = initialize_retriever(args)
    query_embeddings = process_queries(retriever, queries, args.batch_size)

    print("Searching...")
    search_results = search_documents(args, indexes, query_embeddings)
    print("Searching done")

    save_search_results(args, search_results)

if __name__ == '__main__':
    seed_everything(SEED)
    main()