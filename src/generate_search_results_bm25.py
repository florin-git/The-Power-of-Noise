"""
generate_search_results_bm25.py: based on the full corpus as well as the 10k queries dataset, generate the search results using BM25.
                                 As other search results, the output is a file with one entry per query.
                                 Each entry contains a tuple of document ids in the corpus and their BM25 scores.
                                 The user can choose which corpus, queries and the number of top-documents to retrieve.
"""
import argparse
import json
import os
from typing import List, Any, Tuple, Union
import logging

import numpy as np
from bm25s.tokenization import Tokenized
from numpy import ndarray, dtype, signedinteger
import bm25s
from tqdm import tqdm

from utils import write_pickle

LOW_SCORE_PERCENTILE = 0.01
MID_SCORE_RANGE_WIDTH = 0.25 # the range of mid scores is the rango lower-bounded by 0+(0.75/2) and upper-bounded by 1-(0.75/2)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate search results using BM25.')
    parser.add_argument('--corpus_path', type=str, default='data/corpus.json', help='Path to the corpus file.')
    parser.add_argument('--queries_path', type=str, default='data/10k_train_dataset.json', help='Path to the queries file.')
    parser.add_argument('--output_dir', type=str, default='data/search_results/', help='Path to the output directory.')
    parser.add_argument('--k_docs', type=int, default=10, help='Number of documents to retrieve (ex. top k documents or k low-score noise docs). Tuples in search res will have this many docs for each query.')
    parser.add_argument('--op_mode', choices=["top_docs", "low_score_noise", "mid_score_noise"], default='low_score_noise', help='Operating mode of the script: can either generate top docs or low score noise or mid score noise. Default is random.')
    parser.add_argument("--load_idx", type=str, default=None, help="Path to the BM25 index file. Loading it will skip calculate bm25 index and save time. Default is None.")
    parser.add_argument("--save_distrib_figures", action="store_true", help="Save figures of the distribution of BM25 scores for each query. Default is False.")
    parser.add_argument("--num_distrib_figures", type=int, default=10, help="Number of figures to save. Default is 10.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging. Default is False.")
    return parser.parse_args()


def load_corpus(corpus_path: str) -> List[dict]:
    """Load corpus from json_file. Each item in the corpus is an object with keys 'title' and 'text'"""
    corpus = None
    # load json file
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    return corpus


def load_queries(queries_path: str) -> tuple[list[Any], list[Any]]:
    """
    Load queries from dataset. Each sample from train dset is containing:
    example_id: int, question: str, answers: List[str], text: str, gold_document_idx: int
    """
    queries = []
    gold_doc_idxs = []
    json_file = None
    # load json file, keys per sample: ['example_id', 'question', 'answers', 'text', 'idx_gold_in_corpus']
    with open(queries_path, 'r', encoding='utf-8') as f:
        json_file = json.load(f)
    # go over data and get relevant information
    for sample in json_file:
        queries.append(sample['question'])
        gold_doc_idxs.append(sample['idx_gold_in_corpus'])
    return queries, gold_doc_idxs


def preprocess_text(text: str, get_query_token_list=False) -> Union[list[list[str]], Tokenized]:
    """Tokenize and preprocess text for BM25."""
    tokenization_res = bm25s.tokenize(text)
    if get_query_token_list:
        if isinstance(tokenization_res, bm25s.tokenization.Tokenized):
            tokenization_res = bm25s.tokenization.convert_tokenized_to_string_list(tokenization_res)[0]
        else:
            raise ValueError("expected tokenized object, got: ", type(tokenization_res))
    return tokenization_res


def save_search_results(
    args: argparse.Namespace,
    search_results: List[Tuple[List[str], List[float]]],
):
    """Save search results to a pickle file."""
    from pathlib import Path
    os.makedirs(args.output_dir, exist_ok=True)
    outfile_name = f'bm25_{args.op_mode}_{Path(args.queries_path).name}_search_results_at{args.k_docs}.pkl'
    file_path = os.path.join(
        args.output_dir, outfile_name
    )
    print(f"Saving search results to {file_path} ...")
    write_pickle(search_results, file_path)


def get_low_score_noise_documents(scores: List[float], k_docs: int) -> tuple[ndarray[Any, dtype[signedinteger[Any]]], float]:
    """
    Retrieve k low-score noise documents. Return their scores and indices in the corpus.
    """
    # Calculate the threshold for the bottom 20% scores
    threshold = np.quantile(scores, LOW_SCORE_PERCENTILE)

    # Get indices of documents with scores below the threshold
    low_score_indices = np.where(scores <= threshold)[0]

    # The corpus should be large enough so we have enough low score documents to sample
    assert len(low_score_indices) >= k_docs, "Not enough low score documents to sample."
    # Randomly sample k documents from the low score documents
    selected_indices = np.random.choice(low_score_indices, size=k_docs, replace=False)

    # Get the scores for the selected indices
    selected_scores = scores[selected_indices]

    return selected_indices, selected_scores


def get_mid_score_noise_documents(scores: List[float], k_docs: int) -> tuple[
    ndarray[Any, dtype[signedinteger[Any]]], float]:
    """
    Retrieve k low-score noise documents. Return their scores and indices in the corpus.
    """
    # Calculate the threshold for the bottom 20% scores
    min_score = np.min(scores)
    max_score = np.max(scores)
    full_range = max_score - min_score
    current_range = MID_SCORE_RANGE_WIDTH
    mid_score_indices = list()

    while len(mid_score_indices) < k_docs:
        logging.debug(f"Current range: {current_range}, min: {min_score}, max: {max_score}, full range: {full_range}")
        padding = ((1-current_range)/ 2)*full_range
        threshold_low = min_score + padding
        threshold_mid = max_score - padding
        logging.debug(f"Threshold low: {threshold_low}, threshold mid: {threshold_mid}, padding: {padding}")

        # Get indices of documents with scores below the threshold
        gt_low_score_indices = np.where(scores > threshold_low)[0]
        l_scores = scores[gt_low_score_indices]
        mid_score_indices = np.where(l_scores < threshold_mid)[0]
        current_range += 0.1
    logging.debug(f"Found enough mid score indices: {len(mid_score_indices)} with range: {current_range-0.1}")
    # The corpus should be large enough so we have enough low score documents to sample
    assert len(mid_score_indices) >= k_docs, "Not enough low score documents to sample."
    # Randomly sample k documents from the low score documents
    selected_indices = np.random.choice(mid_score_indices, size=k_docs, replace=False)

    # Get the scores for the selected indices
    selected_scores = scores[selected_indices]

    return selected_indices, selected_scores


def compute_bm25_search_results_for_one_query(retriever: bm25s.BM25, op_mode: str, query_text: str, k_docs: int):
    """
    Given a retriever object and a query, compute the BM25 scores for the query and return the top k documents according
    to the op-mode. Assume retriever is already initialized.
    Returns:
    """
    if op_mode == 'top_docs':
        query_tokens = preprocess_text(query_text, get_query_token_list=False)
        scores = None
        top_indices, top_scores = retriever.retrieve(query_tokens, k=k_docs)
    elif op_mode == 'low_score_noise':
        # Get BM25 scores for all documents
        query_tokens = preprocess_text(query_text, get_query_token_list=True)
        scores = retriever.get_scores(query_tokens)
        top_indices, top_scores = get_low_score_noise_documents(scores, k_docs)
    elif op_mode == 'mid_score_noise':
        # Get BM25 scores for all documents
        query_tokens = preprocess_text(query_text, get_query_token_list=True)
        scores = retriever.get_scores(query_tokens)
        top_indices, top_scores = get_mid_score_noise_documents(scores, k_docs)
    else:
        raise ValueError(f"Invalid op_mode: {op_mode}")
    return top_indices, top_scores, scores


def compute_bm25_search_results(
        corpus: List[dict],
        queries: List[str],
        k_docs: int,
        op_mode: str = 'top_docs', # options: top_docs, low_score_noise, mid_score_noise
        store_distribution_figures: bool = False,
        num_distribution_figures: int = 10
) -> List[Tuple[List[str], List[float]]]:
    """
    Compute BM25 search results for the given queries.
    Depending on the op_mode different search results are generated.
    1. top_docs: return top k documents for each query (top score according to bm25)
    2. low_score_noise: return k documents for each query but with low score (randomly select k docs from corpus but with low noise)
    """
    logging.info(f"Compute with op_mode: {op_mode}, k_docs: {k_docs}, store_distribution_figures: {store_distribution_figures}, num_distribution_figures: {num_distribution_figures} ...")
    current_num_distribution_figures = 0
    # initialize tokenizer
    bm25 = initialize_bm25_retriever(args.load_idx, corpus)
    del corpus

    search_results = []
    for query in tqdm(queries, desc="Computing BM25 scores"):

        top_indices, top_scores, scores = compute_bm25_search_results_for_one_query(bm25, op_mode, query, k_docs)

        if scores is not None and store_distribution_figures and current_num_distribution_figures < num_distribution_figures:
            # store score distribution figures
            logging.info(">>> Saving BM25 distribution figures for query: " + query + " <<<")
            logging.info(f"Scores-Max: {np.max(scores)}, Min-{np.min(scores)}")

            # Calculate and log quantiles from 0.1 to 0.9
            quantiles = np.arange(0.1, 1.0, 0.2)
            for q in quantiles:
                logging.info(f"Quantile-{q}: {np.quantile(scores, q)}")
            current_num_distribution_figures += 1
            # Plot the distribution of BM25 scores for the current query
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(style="darkgrid")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(scores, ax=ax)
            ax.set_title(f"BM25 distribution for query: {query}")
            ax.set_xlim(0, np.max(scores))
            ax.set_yscale("log")
            fig.savefig(os.path.join(args.output_dir, f"bm25_distribution_for_query_{query}.png"))
            plt.close(fig)

        # Log debug the scores
        logging.debug(f"BM25 scores for query: {query}")
        logging.debug(top_scores)
        # Convert to strings as per your existing code pattern
        doc_ids = [str(idx) for idx in top_indices]

        search_results.append((doc_ids, top_scores.tolist()))

    return search_results


def initialize_bm25_retriever(load_idx: str, corpus: List[dict]=None):
    """Initialize the BM25 retriever. Either loads the index or computes the index by loading and tokenizing the corpus"""
    if load_idx is not None:
        logging.info("Loading BM25 index from {} ...".format(load_idx))
        bm25 = bm25s.BM25.load(load_idx)
    else:
        assert corpus is not None, "Corpus must be provided if loading index is not possible."
        # Extract text from corpus documents
        logging.info("Extracting text from corpus documents...")
        corpus_texts = [doc.get('text', '') for doc in corpus]

        # Tokenize corpus
        logging.info("Tokenizing corpus...")
        corpus_tokens = bm25s.tokenize(corpus_texts, show_progress=True)

        # Initialize BM25
        logging.info("Initialize BM25 Model...")
        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens)
        # store the index
        bm25.save(os.path.join(args.output_dir, 'bm25_search_results_idx'))
    return bm25


def main(args):
    """Load corpus and queries, compute BM25 search results, and save to file."""
    logging.info("Loading corpus from {} ...".format(args.corpus_path))
    corpus = load_corpus(args.corpus_path)
    logging.info("Loading queries from {}...".format(args.queries_path))
    queries, gold_doc_idxs = load_queries(args.queries_path)
    logging.info("Computing BM25 search results ...")
    search_results = compute_bm25_search_results(corpus, queries, op_mode=args.op_mode, k_docs=args.k_docs, store_distribution_figures=args.save_distrib_figures, num_distribution_figures=args.num_distrib_figures)
    logging.info("Saving search results to {} ...".format(args.search_results_path))
    save_search_results(args, search_results)
    print("Finished computing BM25 search results. Results saved to file.")


if __name__ == "__main__":
    args = parse_args()
    # Set logging level
    log_lvl = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=log_lvl)
    main(args)