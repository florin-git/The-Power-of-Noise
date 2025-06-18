import os
import argparse
import warnings
from tqdm import tqdm
from typing import Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from ColBERTmain.colbert.infra import ColBERTConfig, Run
from ColBERTmain.colbert.modeling.reranker.electra import ElectraReranker
from ColBERTmain.colbert.modeling.reranker.tokenizer import RerankerTokenizer
from src.generate_search_results_bm25 import initialize_bm25_retriever, compute_bm25_search_results_for_one_query
from src.llm import LLM
from src.utils import *
from src.prompt_dataset import PromptDataset, RerankerDataset
from train_colbert_e2e import fake_collate_fn

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED=10

info = {
    "data_path": 'data/10k_train_dataset.json',
    "random_results_path": "data/10k_random_results_at60.pkl",
    "adore_search_results_path": "data/adore_search_results_at200.pkl",
    "contriever_search_results_path": "data/contriever_search_results_at150.pkl",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM Generation with reranker.")
    parser.add_argument('--corpus_path', type=str, default='data/corpus.json', help='Path to the corpus file.')
    parser.add_argument('--output_dir', type=str, default='data/gen_res/reranked', help='Output directory')
    parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='LLM model identifier')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path for the reranker')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length for the LLM model', default=4096)
    parser.add_argument('--load_idx', type=str, help='Load a specific index of the corpus', default=None)
    parser.add_argument('--gold_position', type=int, help='The (0-indexed) position of the gold document in the context')
    parser.add_argument('--num_documents_in_context', type=int, help='Total number of documents in the context')
    parser.add_argument('--noise_type', type=str, default='low_score_noise')
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate', default=15)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_every', type=int, default=250)

    args = parser.parse_args()

    if args.num_documents_in_context is None:
        parser.error("'num_documents_in_context' must be specified.")
    if args.num_documents_in_context <= 0:
        parser.error("'num_documents_in_context' must be a positive integer.")
    if args.gold_position is not None and (args.gold_position < 0 or args.gold_position >= args.num_documents_in_context):
        parser.error("'gold_position' must be within the range of 'num_documents_in_context'.")

    return args


def load_corpus(
        args: argparse.Namespace
) -> Tuple[List[Dict], Optional[Dict[int, int]]]:
    # Load the corpus

    corpus = read_corpus_json(args.corpus_path)
    return corpus, None




def initialize_dataset_and_loader(
        args,
        corpus, full_to_subset_idx_map=None):
    reranker_ds = RerankerDataset(
        corpus=corpus, data_path=info['data_path'],
        do_normalize_query=True,
        num_documents_in_context=args.num_documents_in_context,
        gold_position=args.gold_position,
        full_to_subset_idx_map=full_to_subset_idx_map
    )
    reranker_dataloader = DataLoader(
        reranker_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=fake_collate_fn
    )
    return reranker_dataloader


def print_info(args: argparse.Namespace):
    print("INFO:")
    print(f"DATA: {info['data_path']}")
    print(f"MODEL: {args.llm_id}")
    print(f"GOLD POSITION: {args.gold_position}")
    print(f"NUM DOCUMENTS IN CONTEXT: {args.num_documents_in_context}")
    print(f"BATCH SIZE: {args.batch_size}")
    print(f"SAVE EVERY: {args.save_every}")


def generate_and_save(
        args: argparse.Namespace,
        llm: LLM,
        reranker_dataloader: DataLoader,
        colbert,
        bm25,
        corpus,
        bert_tokenizer
):
    # Info from arguments
    llm_id = args.llm_id
    num_doc = args.num_documents_in_context
    save_every = args.save_every
    gold_pos = args.gold_position

    # Create the saving directory
    llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    saving_dir = f"{args.output_dir}/{llm_folder}/train/classic/reranker/{num_doc}_doc"
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)


    # MPT has a different answer string in the prompt
    answer_string_in_prompt = "### Response:" if 'mpt' in llm_id else "Answer:"

    k_docs = 100

    all_info = []
    for idx, batch in enumerate(tqdm(reranker_dataloader)):
        # query = [b['query'] for b in batch]
        # gold_document = [b['gold_document'] for b in batch]
        # answers = [b['answers'] for b in batch]

        query = [batch['query']]
        gold_document = [batch['gold_document']]
        answers = [batch['answers']]

        for i in range(len(query)):
            idxs, scores, _ = compute_bm25_search_results_for_one_query(bm25, op_mode=args.noise_type, query_text=query[i], k_docs=k_docs)

            passages = [corpus[idx]['text'] for idx in idxs]

            flat_query = [query[i]] * k_docs
            encoding = bert_tokenizer.tensorize(flat_query, passages)
            reranker_scores = colbert(encoding.to(device))

            sorted_indices = torch.argsort(reranker_scores, descending=True).cpu().numpy()

            best_passages = [passages[idx] for idx in sorted_indices[:args.num_documents_in_context-1]]

            best_passages.insert(args.gold_position, gold_document[i])
            documents_str = "\n".join(best_passages)

            task_instruction = "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES."
            prompt = f"""{task_instruction}\nDocuments:\n{documents_str}\nQuestion: {query[i]}"""

            tokens = llm.tokenizer.tokenize(prompt)
            tokens_len = len(tokens)
            if tokens_len >= 4096:  #max_tokenized_length
                print("Skipping example due to prompt length.")
                continue  # Skip adding this example

            generated_out = llm.generate(prompt, max_new_tokens=args.max_new_tokens)
            output = generated_out[0]  # Assuming the output is a list of strings


            start = output.find(answer_string_in_prompt) + len(answer_string_in_prompt)
            response = output[start:].strip()

            # batch[i]['generated_answer'] = response
            # batch[i]['prompt'] = prompt
            batch['generated_answer'] = response
            batch['prompt'] = prompt
            all_info.append(batch)

            if (idx + 1) % save_every == 0 or (idx + 1) == len(reranker_dataloader):
                print(f"Saving at {idx + 1}...")
                file_name = f"{saving_dir}/numdoc{num_doc}_gold_at{gold_pos}_info_{idx+1}.pkl"
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

    print("Loading reranker...")
    config = None
    config = ColBERTConfig.from_existing(config, Run().config)
    checkpoint = args.checkpoint if args.checkpoint else 'google/electra-base-discriminator'
    print(f'Using checkpoint for ElectraReranker: {checkpoint}')
    colbert = ElectraReranker.from_pretrained(checkpoint)
    colbert = colbert.to(device)

    print("Loading corpus...")
    corpus, full_to_subset_idx_map = load_corpus(args)
    print("Corpus loaded")


    print("Loading reranker dataset...")
    prompt_dataloader = initialize_dataset_and_loader(
        args, corpus, full_to_subset_idx_map
    )
    print("Reranker dataset loaded")

    print("Initializing BM25 retriever... ", end ="")
    bm25 = initialize_bm25_retriever(load_idx=args.load_idx)
    print("Done")
    bert_tokenizer = RerankerTokenizer(total_maxlen=config.doc_maxlen, base=checkpoint)

    print_info(args)
    generate_and_save(args, llm, prompt_dataloader, colbert, bm25, corpus, bert_tokenizer)



if __name__ == "__main__":
    seed_everything(SEED)
    main()