import os
import re
import json
import pickle
import argparse

import torch
import pandas as pd
from typing import List, Dict

from utils import str2bool
from normalize_answers import *



def are_answers_matching(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth in normalized_prediction:
            return True
    return False


def read_generation_results(file_path: str, df: pd.DataFrame) -> List[Dict]:
    data = []
    with open(file_path, "r") as fin:
        file_data = json.load(fin)

        for example in file_data:
            example_ids = example['example_id']
            queries = example['query']
            prompts = example['prompt']
            document_indices = list(zip(*example['document_indices']))
            gold_document_indices = example['gold_document_idx']
            generated_answers = example['generated_answer']
            prompt_tokens_lens = example['prompt_tokens_len']

            for i in range(len(example_ids)):
                example_id = example_ids[i]
                query = queries[i]
                gold_document_idx = gold_document_indices[i]
                documents_idx = list(document_indices[i])
                generated_answer = generated_answers[i]
                prompt = prompts[i]
                prompt_tokens_len = prompt_tokens_lens[i]

                answers = df[df['example_id'] == int(example_id)].answers.iloc[0]
                gold_in_retrieved = False

                if int(gold_document_idx) in map(int, documents_idx):
                    gold_in_retrieved = True

                ans_match_after_norm: bool = are_answers_matching(generated_answer, answers)
                ans_in_documents: bool = is_answer_in_text(prompt, answers)
                data.append({
                    'example_id': int(example_id),
                    'query': query,
                    'prompt': prompt,
                    'document_indices': documents_idx,
                    'gold_document_idx': gold_document_idx,
                    'generated_answer': generated_answer,
                    'answers': answers,
                    'ans_match_after_norm': ans_match_after_norm,
                    'gold_in_retrieved': gold_in_retrieved,
                    'ans_in_documents': ans_in_documents,
                    "prompt_tokens_len": prompt_tokens_len,
                })

    return data


def read_generation_results_only_query(file_path: str, df: pd.DataFrame) -> List[Dict]:
    data = []
    with open(file_path, "r") as fin:
        file_data = json.load(fin)

        for example in file_data:
            example_ids = example['example_id']
            queries = example['query']
            prompts = example['prompt']
            generated_answers = example['generated_answer']

            for i in range(len(example_ids)):
                example_id = example_ids[i]
                query = queries[i]
                generated_answer = generated_answers[i]
                prompt = prompts[i]

                answers = df[df['example_id'] == int(example_id)].answers.iloc[0]

                ans_match_after_norm: bool = are_answers_matching(generated_answer, answers)
                ans_in_documents: bool = is_answer_in_text(prompt, answers)
                data.append({
                    'example_id': int(example_id),
                    'query': query,
                    'prompt': prompt,
                    'generated_answer': generated_answer,
                    'answers': answers,
                    'ans_match_after_norm': ans_match_after_norm,
                    'ans_in_documents': ans_in_documents,
                })

    return data


def convert_tensors(cell):
    """ Converts tensors in the given cell to lists, if they are tensors. """
    if isinstance(cell, list):
        return [[t.tolist() if torch.is_tensor(t) else t for t in inner_list] for inner_list in cell]
    return cell


def extract_number_from_filename(filename: str, pattern: re.Pattern) -> int:
    """ Extracts the number from the filename based on the provided pattern. """
    match = pattern.search(filename)
    return int(match.group(1)) if match else 0


def load_pickle_files(directory: str, filename_prefix: str) -> pd.DataFrame:
    """ Loads and concatenates data from all pickle files in the directory with the given prefix. """
    pattern = re.compile(r'(\d+).pkl')
    files = [f for f in os.listdir(directory) if f.endswith('.pkl') and filename_prefix in f]
    files.sort(key=lambda f: extract_number_from_filename(f, pattern))
    print("I'm using the following files: ", files)

    data_list = []
    for file in files:
        with open(os.path.join(directory, file), 'rb') as f:
            data = pickle.load(f)
            data_list.extend(data)
    
    data_df = pd.DataFrame(data_list)
    if 'only_query' in directory:
        data_df['example_id'] = data_df['example_id'].apply(lambda x: x.tolist())
    else:
        data_df['document_indices'] = data_df['document_indices'].apply(convert_tensors)

    if 'prompt_tokens_len' in data_df.columns:
        data_df['prompt_tokens_len'] = data_df['prompt_tokens_len'].apply(lambda x: x.tolist())
    return data_df


def save_data_to_json(data_df: pd.DataFrame, directory: str, filename_prefix: str):
    """ Saves the given DataFrame to a JSON file. """
    data_path = os.path.join(directory, f'{filename_prefix}all.json')
    # Check if the file already exists
    if os.path.exists(data_path):
        overwrite = input(f"File {data_path} already exists. Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("No overwrite.")

            results_df = pd.read_json(f'{directory}/{filename_prefix}all_extended.json')
            accuracy = round(results_df['ans_match_after_norm'].sum() / len(results_df), 4)
            print("ACCURACY: ", accuracy)
            return None
        
    data_df.to_json(data_path, orient='records')
    return data_path


def get_classic_path(args):
    gold_pos = args.gold_position
    rand_str = "_rand" if args.use_random else ""
    answerless_str = "_answerless" if args.get_documents_without_answer else ""
    adore_str = "_adore" if args.use_adore else ""

    filename_prefix = f'numdoc{args.num_doc}_gold_at{gold_pos}{rand_str}{answerless_str}{adore_str}_info_'
    return filename_prefix


def get_reranker_path(args):
    gold_pos = args.gold_position
    rand_str = "_rand" if args.use_random else ""
    answerless_str = "_answerless" if args.get_documents_without_answer else ""
    adore_str = "_adore" if args.use_adore else ""

    filename_prefix = f'numdoc{args.num_doc}_gold_at{gold_pos}{rand_str}{answerless_str}{adore_str}_info_'
    return filename_prefix


def get_mixed_path(args):
    answerless_str = "_answerless" if args.get_documents_without_answer else ""

    if args.put_retrieved_first:
        first_type_str = f"_retr{args.num_retrieved_documents}"
        second_type_str = f"_rand{args.num_random_documents}"
    else:
        first_type_str = f"_rand{args.num_random_documents}"
        second_type_str = f"_retr{args.num_retrieved_documents}"
    
    filename_prefix = f"numdoc{args.num_doc}{first_type_str}{second_type_str}{answerless_str}_info_"
    return filename_prefix


def get_multi_corpus_path(args):
    answerless_str = "_answerless" if args.get_documents_without_answer else ""
    other_corpus_str = "_nonsense" if args.use_corpus_nonsense else "_reddit"

    if args.put_main_first:
        first_type_str = f"_main{args.num_main_documents}"
        second_type_str = f"_other{args.num_other_documents}"
    else:
        first_type_str = f"_other{args.num_other_documents}"
        second_type_str = f"_main{args.num_main_documents}"
    
    filename_prefix = f"numdoc{args.num_doc}{first_type_str}{second_type_str}{answerless_str}{other_corpus_str}_info_"
    return filename_prefix


def get_only_query_path():
    filename_prefix = f"only_query_info_"
    return filename_prefix


def parse_arguments():
    parser = argparse.ArgumentParser(description="Read Generation Results.")
    parser.add_argument('--output_dir', type=str, default='data/gen_res', help='Output directory')
    parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='LLM model identifier')
    parser.add_argument('--use_test', type=str2bool, help='Use the test set')
    parser.add_argument('--prompt_type', type=str, default='classic', help='Which type of prompt to use [classic, mixed, multi_corpus, only_query]')
    
    parser.add_argument('--use_random', type=str2bool, help='Use random irrelevant documents')
    parser.add_argument('--use_adore', type=str2bool, help="Use the retrieved documents from ADORE")
    parser.add_argument('--gold_position', type=int, help='The (0-indexed) position of the gold document in the context')
    parser.add_argument('--num_documents_in_context', type=int, help='Total number of documents in the context')
    parser.add_argument('--get_documents_without_answer', type=str2bool, help='Select only documents without the answer (e.g., distracting)')

    parser.add_argument('--use_bm25', type=str2bool, help="Use the retrieved documents from BM25")
    parser.add_argument('--num_retrieved_documents', type=int, help='Number of retrieved documents in the context')
    parser.add_argument('--num_random_documents', type=int, help='Number of random documents in the context')
    parser.add_argument('--put_retrieved_first', type=str2bool, help='Put the retrieved documents first in the context')

    parser.add_argument('--use_corpus_nonsense', type=str2bool, help="Use documents composed of random words")
    parser.add_argument('--num_main_documents', type=int, help='Number of documents in the context from the main corpus')
    parser.add_argument('--num_other_documents', type=int, help='Number of documents in the context from the other corpus')
    parser.add_argument('--put_main_first', type=str2bool, help='Put the documents of the main corpus first in the context')
    

    args = parser.parse_args()

    if not args.prompt_type in ['classic', 'mixed', 'multi_corpus', 'only_query', 'reranker']:
        parser.error("Invalid prompt type. Must be one of ['classic', 'mixed', 'multi_corpus', 'only_query']")

    return args

def main():
    args = parse_arguments()
    
    retriever_str = ""
    
    prompt_type = args.prompt_type
    if prompt_type == 'classic':
        retriever_str = "adore/" if args.use_adore else "contriever/"
        args.num_doc = args.num_documents_in_context
        filename_prefix = get_classic_path(args)
    elif prompt_type == 'reranker':
        retriever_str = "adore/" if args.use_adore else "contriever/"
        args.num_doc = args.num_documents_in_context
        filename_prefix = get_reranker_path(args)
    elif prompt_type == 'mixed':
        retriever_str = "bm25/" if args.use_bm25 else "contriever/"
        args.num_doc = args.num_retrieved_documents + args.num_random_documents
        filename_prefix = get_mixed_path(args)
    elif prompt_type == 'multi_corpus':
        retriever_str = "bm25/" if args.use_bm25 else "contriever/"
        args.num_doc = args.num_main_documents + args.num_other_documents
        filename_prefix = get_multi_corpus_path(args)
    elif prompt_type == 'only_query':
        filename_prefix = get_only_query_path()
    else:
        raise ValueError("Invalid prompt type")


    llm_id = args.llm_id
    split = "test" if args.use_test else "train"
    llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    doc_str = f"{args.num_doc}_doc" if 'only_query' not in prompt_type else ""
    directory = f'{args.output_dir}/{llm_folder}/{split}/{prompt_type}/{retriever_str}{doc_str}'
    print("Directory: ", directory)

    data_df = load_pickle_files(directory, filename_prefix)
    data_path = save_data_to_json(data_df, directory, filename_prefix)
    if data_path is None:
        return
    

    if args.use_test:
        df = pd.read_json('data/test_dataset.json')
    else:
        df = pd.read_json('data/10k_train_dataset.json')

    if 'only_query' in directory:
        results = read_generation_results_only_query(data_path, df)
    else:
        results = read_generation_results(data_path, df)

    results_df = pd.DataFrame(results)
    accuracy = round(results_df['ans_match_after_norm'].sum() / len(results_df), 4)
    print("ACCURACY: ", accuracy)
    results_df.to_json(os.path.join(directory, f'{filename_prefix}all_extended.json'), orient='records')


if __name__ == "__main__":
    main()
