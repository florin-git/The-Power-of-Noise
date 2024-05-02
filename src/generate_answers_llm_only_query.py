import os 
import argparse
import warnings
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from llm import LLM
from utils import *
from prompt_dataset import QueryDataset


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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run LLM Closed-Book Generation (only query).")
    parser.add_argument('--output_dir', type=str, default='data/gen_res', help='Output directory')
    parser.add_argument('--llm_id', type=str, default='meta-llama/Llama-2-7b-chat-hf', help='LLM model identifier')
    parser.add_argument('--model_max_length', type=int, help='Maximum input length for the LLM model', default=4096)
    parser.add_argument('--use_test', type=str2bool, help='Use the test set', default=False)
    parser.add_argument('--max_new_tokens', type=int, help='Maximum number of tokens to generate', default=15)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_every', type=int, default=250)

    args = parser.parse_args()
    args.split = "test" if args.use_test else "train"
    return args


def initialize_dataset_and_loader(
    args: argparse.Namespace, 
) -> DataLoader:
    
    prompt_ds = QueryDataset(
        data_path=info[args.split]['data_path'],
        model_name=args.llm_id, 
        do_normalize_query=True, 
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
    print("ONLY QUERY")
    print(f"DATA: {info[args.split]['data_path']}")
    print(f"MODEL: {args.llm_id}")
    print(f"BATCH SIZE: {args.batch_size}")
    print(f"SAVE EVERY: {args.save_every}")


def generate_and_save(
    args: argparse.Namespace, 
    llm: LLM, 
    prompt_dataloader: DataLoader
):
    # Info from arguments
    llm_id = args.llm_id
    save_every = args.save_every

    # Create the saving directory
    llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    saving_dir = f"{args.output_dir}/{llm_folder}/{args.split}/only_query"
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
            file_name = f"{saving_dir}/only_query_info_{idx+1}.pkl"
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
    print("LLM loaded")

    print("Loading prompt dataset...")
    prompt_dataloader = initialize_dataset_and_loader(args)
    print("Prompt dataset loaded")

    print_info(args)
    generate_and_save(args, llm, prompt_dataloader)



if __name__ == '__main__':
    seed_everything(SEED)
    main()