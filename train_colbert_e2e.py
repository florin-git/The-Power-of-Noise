import argparse
import os
import time
import pickle
from typing import List

from src.normalize_answers import *

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from ColBERTmain.colbert.infra import ColBERTConfig, Run
from ColBERTmain.colbert.modeling.reranker.electra import ElectraReranker
from ColBERTmain.colbert.training.rerank_batcher import RerankBatcher
from ColBERTmain.colbert.training.training import set_bert_grad
from ColBERTmain.colbert.utils.amp import MixedPrecisionManager
from ColBERTmain.colbert.modeling.reranker.tokenizer import RerankerTokenizer
from src.generate_search_results_bm25 import initialize_bm25_retriever, compute_bm25_search_results_for_one_query
from src.llm import LLM
from src.prompt_dataset import PromptDataset, RerankerDataset
from src.utils import str2bool, read_corpus_json, read_corpus_with_contriever

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

info = {
    "data_path": 'data/10k_train_dataset.json',
    "random_results_path": "data/10k_random_results_at60.pkl",
    "adore_search_results_path": "data/adore_search_results_at200.pkl",
    "contriever_search_results_path": "data/contriever_search_results_at150.pkl",
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM Generation.")
    parser.add_argument('--output_dir', type=str, default='data/gen_res/bert', help='Output directory')
    parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length for the LLM model', default=4096)
    parser.add_argument('--load_full_corpus', type=str2bool, help='Load the full corpus', default=True)
    parser.add_argument('--load_idx', type=str, help='Load a specific index of the corpus', default=None)
    parser.add_argument('--gold_position', type=int, help='The (0-indexed) position of the gold document in the context')
    parser.add_argument('--noise_type', type=str, default='low_score_noise')
    parser.add_argument('--num_documents_in_context', type=int, help='Total number of documents in the context')
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate', default=15)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_every', type=int, default=250)
    parser.add_argument('--use_adore', type=str2bool, help="Use the retrieved documents from ADORE", default=False)

    args = parser.parse_args()

    if args.num_documents_in_context is None:
        parser.error("'num_documents_in_context' must be specified.")
    if args.num_documents_in_context <= 0:
        parser.error("'num_documents_in_context' must be a positive integer.")
    if args.gold_position is not None and (args.gold_position < 0 or args.gold_position >= args.num_documents_in_context):
        parser.error("'gold_position' must be within the range of 'num_documents_in_context'.")

    return args

def fake_collate_fn(batch):
    """
    A fake collate function that does nothing.
    This is used to avoid errors when the DataLoader tries to collate the data.
    """
    return batch


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

def read_corpus_with_random():
    full_to_subset_path = "data/mappings/full_to_subset_random_at60_in_corpus.pkl"
    subset_to_full_path = "data/mappings/subset_to_full_random_at60_in_corpus.pkl"
    corpus_path = "data/processed/corpus_with_random_at60.json"
    return read_subset_corupus_with_map(
        full_to_subset_path,
        subset_to_full_path,
        corpus_path
    )


def load_corpus(args: argparse.Namespace) :
    # Load the corpus
    if args.load_full_corpus:
        print("Loading full corpus from JSON file...")
        corpus = read_corpus_json('data/corpus.json')
        return corpus, None

    # if args.use_random:
    #     corpus, full_to_subset_idx_map = read_corpus_with_random()
    if args.use_adore: #elif
        print("Loading corpus with ADORE search results...")
        corpus, full_to_subset_idx_map = read_corpus_with_adore()
    else:
        # Corpus with documents from Contriever
        print("Loading corpus with Contriever search results...")
        corpus, full_to_subset_idx_map = read_corpus_with_contriever()

    return corpus, full_to_subset_idx_map

def read_pickle(file_path: str):
    with open(file_path, "rb") as reader:
        data = pickle.load(reader)
    return data

def read_subset_corupus_with_map(
        full_to_subset_path: str,
        subset_to_full_path: str,
        corpus_path: str):
    full_to_subset_idx_map = read_pickle(full_to_subset_path)
    subset_to_full_idx_map = read_pickle(subset_to_full_path)
    corpus = read_corpus_json(corpus_path, subset_to_full_idx_map)
    return corpus, full_to_subset_idx_map

def read_corpus_with_adore():
    full_to_subset_path = "data/mappings/full_to_subset_adore_at200_in_corpus.pkl"
    subset_to_full_path = "data/mappings/subset_to_full_adore_at200_in_corpus.pkl"
    corpus_path = "data/processed/corpus_with_adore_at200.json"
    return read_subset_corupus_with_map(
        full_to_subset_path,
        subset_to_full_path,
        corpus_path
    )

def save_model(args, colbert, optimizer, idx, savepath=None):
    checkpoints_path = savepath or os.path.join(Run().path_, 'checkpoints')
    name = None
    print(checkpoints_path)

    try:
        save = colbert.save
    except:
        save = colbert.module.save

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    if idx != -1:
        path_save = os.path.join(checkpoints_path, f"colbert-{idx}")
    else:
        path_save = os.path.join(checkpoints_path, "colbert_last")
    save(path_save)
    print(f"Saved checkpoint to {path_save}")


def train(config: ColBERTConfig, triples, queries=None, collection=None):
    # reader = RerankBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)

    colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(device)
    colbert.train()

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    scheduler = None

    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)

    args = parse_arguments()

    print("Loading LLM... ")
    llm_id = args.llm_id
    llm = LLM(
        llm_id, device, quantization_bits=4,
        model_max_length=args.model_max_length
    )
    bert_tokenizer = RerankerTokenizer(total_maxlen=config.doc_maxlen, base=config.checkpoint)
    llm_tokenizer = llm.tokenizer

    print("Loading corpus and search results... ")
    corpus, full_to_subset_idx_map = load_corpus(args)
    print("Done")

    reranker_dataloader = initialize_dataset_and_loader(
        args, corpus, full_to_subset_idx_map
    )

    print("Initializing BM25 retriever... ", end ="")
    bm25 = initialize_bm25_retriever(load_idx=args.load_idx)
    print("Done")

    k_docs = 100

    for idx, batch in enumerate(tqdm(reranker_dataloader)):

        query = [b['query'] for b in batch]
        gold_document = [b['gold_document'] for b in batch]
        answers = [b['answers'] for b in batch]

        for i in range(len(query)):
            idxs, scores, _ = compute_bm25_search_results_for_one_query(bm25, op_mode=args.noise_type, query_text=query[i], k_docs=k_docs)

            passages = [corpus[idx]['text'] for idx in idxs]

            flat_query = [query[i]] * k_docs
            encoding = bert_tokenizer.tensorize(flat_query, passages)
            reranker_scores = colbert(encoding.to(device))

            sorted_indices = torch.argsort(reranker_scores, descending=True).cpu().numpy()

            probs = torch.softmax(reranker_scores, dim=-1)
            top_num_context = sorted_indices[:args.num_documents_in_context-1]
            log_probs = torch.log(probs[top_num_context])

            best_passages = [passages[idx] for idx in sorted_indices[:args.num_documents_in_context-1]]


            best_passages.insert(args.gold_position, gold_document[i])
            documents_str = "\n".join(best_passages)

            task_instruction = "You are given a question and you MUST respond by EXTRACTING the answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES."
            prompt = f"""{task_instruction}\nDocuments:\n{documents_str}\nQuestion: {query[i]}\nAnswer:"""

            tokens = llm_tokenizer.tokenize(prompt)
            tokens_len = len(tokens)
            if tokens_len >= 4096:  #max_tokenized_length
                print("Skipping example due to prompt length.")
                continue  # Skip adding this example

            generated_out = llm.generate(prompt, max_new_tokens=args.max_new_tokens)
            generated_output = generated_out[0]  # Assuming the output is a list of strings

            answer_string_in_prompt = "### Response:" if 'mpt' in llm_id else "Answer:"

            if answer_string_in_prompt not in generated_output:
                print(f"0 loss due to missing answer string in output")
                ans_match_after_norm = False
            else:
                start = generated_output.find(answer_string_in_prompt) + len(answer_string_in_prompt)
                response = generated_output[start:].strip()

                ans_match_after_norm: bool = are_answers_matching(response, answers[i])

            reward = 1.0 if ans_match_after_norm else 0.0
            loss = -reward * log_probs.sum()
            # loss.backward()
            amp.backward(loss)
        amp.step(colbert, optimizer, scheduler)
        if idx % 1000 == 0 and idx != 0:
            save_model(config, colbert, optimizer, idx, savepath='/mnt/2tb-1/louis/colbert')

    save_model(config, colbert, optimizer, -1, savepath='/mnt/2tb-1/louis/colbert')




def are_answers_matching(prediction: str, ground_truths: List[str]) -> bool:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth in normalized_prediction:
            return True
    return False


def is_answer_in_text(text: str, answers: List[str]) -> bool:
    """
    Checks if any of the provided answers are present in the given text after normalization.
    """
    for a in answers:
        normalized_answer_lower = normalize_answer(a, lowercase=True)
        normalized_answer = normalize_answer(a, lowercase=False)
        normalized_text = white_space_fix(remove_punc(text))

        if (a in text or
                normalized_answer_lower in normalized_text or
                normalized_answer in normalized_text):
            return True

    return False

def flatten(L):
    # return [x for y in L for x in y]

    result = []
    for _list in L:
        result += _list

    return result

if __name__ == "__main__":
    config = None
    config = ColBERTConfig.from_existing(config, Run().config)
    triples = None
    queries = None
    collection = None
    checkpoint = 'google/electra-base-discriminator'

    config.configure(triples=triples, queries=queries, collection=collection)
    config.configure(checkpoint=checkpoint)

    train(config, triples, queries, collection)

