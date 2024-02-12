import os 
import argparse
import warnings
from tqdm import tqdm
from typing import Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from llm import LLM
from utils import *
from prompt_dataset import MultiCorpusDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED=10

info = {
    "train": {
        "data_path": 'data/10k_train_dataset.json',
        "random_results_path": "data/10k_random_results_at60.pkl",
        "contriever_search_results_path": "data/contriever_search_results_at150.pkl",
    },
    "test": {
        "data_path": 'data/test_dataset.json',
        "contriever_search_results_path": "data/contriever_test_search_results_at150.pkl",
        "bm25_search_results_path": "data/bm25_test_search_results_at250.pkl",
        "nonsense_results_path": "data/nonsense_random_results.pkl",
        "reddit_results_path": "data/reddit_test_random_results.pkl",
    }
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM Generation with multi corpus documents.")
    parser.add_argument('--output_dir', type=str, default='data/gen_res', help='Output directory')
    parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length for the LLM model', default=4096)
    parser.add_argument('--load_full_corpus', type=str2bool, help='Load the full corpus', default=True)    
    parser.add_argument('--use_bm25', type=str2bool, help="Use the retrieved documents from BM25", default=False)
    parser.add_argument('--use_corpus_nonsense', type=str2bool, help="Use documents composed of random words", default=False)
    parser.add_argument('--gold_position', type=int, help='The (0-indexed) position of the gold document in the context', default=None)
    parser.add_argument('--num_main_documents', type=int, help='Number of documents in the context from the main corpus')
    parser.add_argument('--num_other_documents', type=int, help='Number of documents in the context from the other corpus')
    parser.add_argument('--put_main_first', type=str2bool, help='Put the documents of the main corpus first in the context', default=False)
    parser.add_argument('--get_documents_without_answer', type=str2bool, help='Select only documents without the answer (e.g., related)', default=False)
    parser.add_argument('--use_test', type=str2bool, help='Use the test set', default=True)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_every', type=int, default=250)

    args = parser.parse_args()
    args.split = "test" if args.use_test else "train"
    args.num_documents_in_context = args.num_main_documents + args.num_other_documents


    if args.num_main_documents is None or args.num_other_documents is None:
        parser.error("'num_main_documents' and 'num_other_documents' must be specified")
    if args.num_main_documents <= 0 or args.num_other_documents <= 0:
        parser.error("'num_main_documents' and 'num_other_documents' must be a positive integer.")

    return args


def load_corpus(
    args: argparse.Namespace
) -> Tuple[List[Dict], Optional[Dict[int, int]]]:
    # Load other corpus
    if args.use_corpus_nonsense:
        # Corpus with random words
        other_corpus = read_pickle('data/processed/corpus_with_random_50_words.pkl')
    else:
        # Reddit corpus
        other_corpus = read_pickle('data/processed/reddit_corpus.pkl')


    # Load main corpus
        
    if args.load_full_corpus:
        corpus = read_corpus_json('data/corpus.json')
        return corpus, None, other_corpus

    if args.use_test:
        # Use BM25 test corpus or Contriever and random test corpus based on flags
        if args.use_bm25:
            corpus_loader = read_test_corpus_with_random_and_bm25
        else:
            corpus_loader = read_test_corpus_with_random_and_contriever
    else:
        # Default to train corpus with Contriever and random
        corpus_loader = read_corpus_with_random_and_contriever
    
    corpus, full_to_subset_idx_map = corpus_loader()
    return corpus, full_to_subset_idx_map, other_corpus
    

def load_search_results(args: argparse.Namespace) -> List[Tuple[List[int], List[float]]]:
    if args.use_bm25:
        search_results_path = info[args.split]['bm25_search_results_path']
    else:
        search_results_path = info[args.split]['contriever_search_results_path']
    search_results = read_pickle(search_results_path)

    if args.use_corpus_nonsense:
        # Random from nonsensical documents 
        search_results_other_corpus_path = info[args.split]['nonsense_results_path']
    else:
        # Random from Reddit
        search_results_other_corpus_path = info[args.split]['reddit_results_path']
    search_results_other_corpus = read_pickle(search_results_other_corpus_path)

    return search_results, search_results_other_corpus


def initialize_dataset_and_loader(
    args: argparse.Namespace, 
    corpus: List[Dict], 
    other_corpus: List[Dict],
    full_to_subset_idx_map: Optional[Dict[int, int]], 
    search_results: List[Tuple[List[int], List[float]]], 
    search_results_other_corpus: List[Tuple[List[int], List[float]]], 
    tokenizer: PreTrainedTokenizer
) -> DataLoader:
    
    documents_disposition_info = {
        "num_main_documents": args.num_main_documents,
        "num_other_documents": args.num_other_documents,
        "put_main_first": args.put_main_first,
    }
    prompt_ds = MultiCorpusDataset(
        corpus=corpus, data_path=info[args.split]['data_path'], 
        tokenizer=tokenizer, 
        max_tokenized_length=args.model_max_length - 2, 
        search_results=search_results,
        documents_other_corpus=other_corpus,
        search_results_other_corpus=search_results_other_corpus, 
        documents_disposition_info=documents_disposition_info,
        full_to_subset_idx_map=full_to_subset_idx_map,
        do_normalize_query=True, 
        gold_position=args.gold_position, # None in these experiments
        get_documents_without_answer=args.get_documents_without_answer,
    )
        
    prompt_dataloader = DataLoader(
        prompt_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return prompt_dataloader


def print_info(args: argparse.Namespace):
    print("INFO:")
    print("MULTI CORPUS")
    print(f"DATA: {info[args.split]['data_path']}")
    print(f"USE TEST: {args.use_test}")
    print(f"MODEL: {args.llm_id}")
    print(f"USE BM25: {args.use_bm25}")
    print(f"USE CORPUS WITH NONSENSE: {args.use_corpus_nonsense}")
    print(f"NUM MAIN DOCS: {args.num_main_documents}")
    print(f"NUM OTHER DOCS: {args.num_other_documents}")
    print(f"PUT MAIN DOCS FIRST: {args.put_main_first}")
    print(f"GOLD POSITION: {args.gold_position}")
    print(f"NUM DOCUMENTS IN CONTEXT: {args.num_documents_in_context}")
    print(f"DOCUMENTS WITHOUT ANSWER: {args.get_documents_without_answer}")
    print(f"BATCH SIZE: {args.batch_size}")
    print(f"SAVE EVERY: {args.save_every}")


def generate_and_save(
    args: argparse.Namespace, 
    llm: LLM, 
    prompt_dataloader: DataLoader
):
    # Info from arguments
    llm_id = args.llm_id
    num_doc = args.num_documents_in_context
    save_every = args.save_every
    answerless_str = "_answerless" if args.get_documents_without_answer else ""
    retriever_str = "bm25" if args.use_bm25 else "contriever"
    other_corpus_str = "_nonsense" if args.use_corpus_nonsense else "_reddit"

    if args.put_main_first:
        first_type_str = f"_main{args.num_main_documents}"
        second_type_str = f"_other{args.num_other_documents}"
    else:
        first_type_str = f"_other{args.num_other_documents}"
        second_type_str = f"_main{args.num_main_documents}"


    # Create the saving directory
    llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    saving_dir = f"{args.output_dir}/{llm_folder}/{args.split}/multi_corpus/{retriever_str}/{num_doc}_doc"
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    
    # MPT has a different answer string in the prompt
    answer_string_in_prompt = "### Response:" if 'mpt' in llm_id else "Answer:"

    all_info = []  
    for idx, prompt_batch in enumerate(tqdm(prompt_dataloader)):
        prompts = prompt_batch['prompt']
        generated_output = llm.generate(prompts)
        
        generated_answers = []
        for output in generated_output:
            start = output.find(answer_string_in_prompt) + len(answer_string_in_prompt)
            response = output[start:].strip()
            generated_answers.append(response)

        prompt_batch['generated_answer'] = generated_answers
        all_info.append(prompt_batch)
        
        if (idx + 1) % save_every == 0 or (idx + 1) == len(prompt_dataloader):
            print(f"Saving at {idx + 1}...")
            file_name = f"{saving_dir}/numdoc{num_doc}{first_type_str}{second_type_str}{answerless_str}{other_corpus_str}_info_{idx+1}.pkl"
            write_pickle(all_info, file_name)
            all_info = []


def main():
    args = parse_arguments()

    print("Loading LLM...")
    llm_id = args.llm_id
    llm = LLM(
        llm_id, device, quantization_bits=4, 
        model_max_length=args.model_max_length
    )
    tokenizer = llm.tokenizer
    print("LLM loaded")


    print("Loading corpus and search results...")
    corpus, full_to_subset_idx_map, other_corpus = load_corpus(args)
    search_results, search_results_other_corpus = load_search_results(args)
    print("Corpus and search results loaded")


    print("Loading prompt dataset...")
    prompt_dataloader = initialize_dataset_and_loader(
        args, corpus, other_corpus, full_to_subset_idx_map, 
        search_results, search_results_other_corpus, tokenizer
    )
    print("Prompt dataset loaded")

    print_info(args)
    generate_and_save(args, llm, prompt_dataloader)



if __name__ == "__main__":
    SEED=10
    seed_everything(SEED)
    main()