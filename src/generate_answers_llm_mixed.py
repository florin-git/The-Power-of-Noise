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
from prompt_dataset import MixedDocumentsDataset


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
        "random_results_path": "data/10k_other_random_results_at60.pkl",
        "contriever_search_results_path": "data/contriever_test_search_results_at150.pkl",
        "bm25_search_results_path": "data/bm25_test_search_results_at250.pkl",
    }
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM Generation with mixed documents (retrieved and random).")
    parser.add_argument('--output_dir', type=str, default='data/gen_res', help='Output directory')
    parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length for the LLM model', default=4096)
    parser.add_argument('--load_full_corpus', type=str2bool, help='Load the full corpus', default=True)    
    parser.add_argument('--use_bm25', type=str2bool, help="Use the retrieved documents from BM25", default=False)
    parser.add_argument('--gold_position', type=int, help='The (0-indexed) position of the gold document in the context', default=None)
    parser.add_argument('--num_retrieved_documents', type=int, help='Number of retrieved documents in the context')
    parser.add_argument('--num_random_documents', type=int, help='Number of random documents in the context')
    parser.add_argument('--put_retrieved_first', type=str2bool, help='Put the retrieved documents first in the context', default=False)
    parser.add_argument('--get_documents_without_answer', type=str2bool, help='Select only documents without the answer (e.g., distracting)', default=False)
    parser.add_argument('--use_test', type=str2bool, help='Use the test set', default=True)
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate', default=15)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_every', type=int, default=250)

    args = parser.parse_args()
    args.split = "test" if args.use_test else "train"
    args.num_documents_in_context = args.num_retrieved_documents + args.num_random_documents


    if args.num_retrieved_documents is None or args.num_random_documents is None:
        parser.error("'num_retrieved_documents' and 'num_random_documents' must be specified")
    if args.num_retrieved_documents <= 0 and args.num_random_documents <= 0:
        parser.error("'num_retrieved_documents' and 'num_random_documents' must not both be zero or negative.")

    return args


def load_corpus(
    args: argparse.Namespace
) -> Tuple[List[Dict], Optional[Dict[int, int]]]:
    # Load the corpus
    if args.load_full_corpus:
        corpus = read_corpus_json('data/corpus.json')
        return corpus, None

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
    return corpus, full_to_subset_idx_map


def load_search_results(args: argparse.Namespace) -> List[Tuple[List[int], List[float]]]:
    random_results_path = info[args.split]['random_results_path']
    random_search_results = read_pickle(random_results_path)

    if args.use_bm25:
        search_results_path = info[args.split]['bm25_search_results_path']
    else:
        search_results_path = info[args.split]['contriever_search_results_path']
    retriever_search_results = read_pickle(search_results_path)

    return retriever_search_results, random_search_results


def initialize_dataset_and_loader(
    args: argparse.Namespace, 
    corpus: List[Dict], 
    full_to_subset_idx_map: Optional[Dict[int, int]], 
    retriever_search_results: List[Tuple[List[int], List[float]]], 
    random_search_results: List[Tuple[List[int], List[float]]], 
    tokenizer: PreTrainedTokenizer
) -> DataLoader:
    
    documents_disposition_info = {
        "num_retrieved_documents": args.num_retrieved_documents,
        "num_random_documents": args.num_random_documents,
        "put_retrieved_first": args.put_retrieved_first,
    }
    prompt_ds = MixedDocumentsDataset(
        corpus=corpus, data_path=info[args.split]['data_path'], 
        tokenizer=tokenizer, 
        max_tokenized_length=args.model_max_length - 2, 
        retriever_search_results=retriever_search_results,
        random_search_results=random_search_results,
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
    print("MIXED")
    print(f"DATA: {info[args.split]['data_path']}")
    print(f"USE TEST: {args.use_test}")
    print(f"MODEL: {args.llm_id}")
    print(f"USE BM25: {args.use_bm25}")
    print(f"NUM RETRIEVED DOCS: {args.num_retrieved_documents}")
    print(f"NUM RANDOM DOCS: {args.num_random_documents}")
    print(f"PUT RETRIEVED DOCS FIRST: {args.put_retrieved_first}")
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

    if args.put_retrieved_first:
        first_type_str = f"_retr{args.num_retrieved_documents}"
        second_type_str = f"_rand{args.num_random_documents}"
    else:
        first_type_str = f"_rand{args.num_random_documents}"
        second_type_str = f"_retr{args.num_retrieved_documents}"


    # Create the saving directory
    llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    saving_dir = f"{args.output_dir}/{llm_folder}/{args.split}/mixed/{retriever_str}/{num_doc}_doc"
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    
    # MPT has a different answer string in the prompt
    answer_string_in_prompt = "### Response:" if 'mpt' in llm_id else "Answer:"

    all_info = []  
    for idx, prompt_batch in enumerate(tqdm(prompt_dataloader)):
        prompts = prompt_batch['prompt']
        generated_output = llm.generate(prompts, max_new_tokens=args.max_new_tokens)
        
        generated_answers = []
        for output in generated_output:
            start = output.find(answer_string_in_prompt) + len(answer_string_in_prompt)
            response = output[start:].strip()
            generated_answers.append(response)

        prompt_batch['generated_answer'] = generated_answers
        all_info.append(prompt_batch)
        
        if (idx + 1) % save_every == 0 or (idx + 1) == len(prompt_dataloader):
            print(f"Saving at {idx + 1}...")
            file_name = f"{saving_dir}/numdoc{num_doc}{first_type_str}{second_type_str}{answerless_str}_info_{idx+1}.pkl"
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
    corpus, full_to_subset_idx_map = load_corpus(args)
    retriever_search_results, random_search_results = load_search_results(args)
    print("Corpus and search results loaded")


    print("Loading prompt dataset...")
    prompt_dataloader = initialize_dataset_and_loader(
        args, corpus, full_to_subset_idx_map, retriever_search_results, 
        random_search_results, tokenizer
    )
    print("Prompt dataset loaded")

    print_info(args)
    generate_and_save(args, llm, prompt_dataloader)



if __name__ == "__main__":
    seed_everything(SEED)
    main()