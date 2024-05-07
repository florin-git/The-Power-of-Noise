import os 
import argparse
import warnings

import torch
from transformers import AutoTokenizer, AutoConfig

from retriever import *
from utils import *


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED=10


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script for computing the embeddings of a corpus.")
    parser.add_argument('--corpus_path', type=str, help='Path to the JSON corpus data')
    parser.add_argument('--encoder_id', type=str, default='facebook/contriever', help='Model identifier for the encoder')
    parser.add_argument('--max_length_encoder', type=int, default=512, help='Maximum sequence length for the encoder')
    parser.add_argument('--normalize_embeddings', type=str2bool, default=False, help='Whether to normalize embeddings')
    parser.add_argument('--lower_case', type=str2bool, default=False, help='Whether to lower case the corpus text')
    parser.add_argument('--do_normalize_text', type=str2bool, default=True, help='Whether to normalize the corpus text')
    parser.add_argument('--output_dir', type=str, default='data/corpus/embeddings/', help='Output directory for saving embeddings')
    parser.add_argument('--prefix_name', type=str, default='contriever', help='Initial part of the name of the saved embeddings')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for encoding queries')
    parser.add_argument('--save_every', type=int, default=500)
    
    args =  parser.parse_args()

    return args


def initialize_retriever(args: argparse.Namespace) -> Retriever:
    """Initialize the encoder and retriever."""
    config = AutoConfig.from_pretrained(args.encoder_id)
    encoder = Encoder(config).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_id)
    retriever = Retriever(
        device=device, tokenizer=tokenizer, 
        query_encoder=encoder, 
        max_length=args.max_length_encoder,
        norm_doc_emb=args.normalize_embeddings,
        lower_case=args.lower_case,
        do_normalize_text=args.do_normalize_text
    )

    return retriever


def main():
    args = parse_arguments()

    print("Loading corpus...")
    corpus = read_json(args.corpus_path)
    print("Corpus loaded")

    retriever = initialize_retriever(args)
    print("Computing embeddings...")
    retriever.encode_corpus(
        corpus, 
        batch_size=args.batch_size, 
        output_dir=args.output_dir,
        prefix_name=args.prefix_name,
        save_every=args.save_every
    )

if __name__ == '__main__':
    seed_everything(SEED)
    main()