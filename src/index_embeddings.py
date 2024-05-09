import os 
import argparse
import warnings

from index import *
from utils import *

warnings.filterwarnings('ignore')
SEED=10


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script for computing the embeddings of a corpus.")
    parser.add_argument('--corpus_size', type=int, help='Size of the embedded corpus')
    parser.add_argument('--vector_sz', type=int, default=768, help='Size of the vectors to be indexed')
    parser.add_argument('--idx_type', type=str, default='IP', help='Index type (IP for Inner Product)')
    parser.add_argument('--faiss_dir', type=str, help='Directory where to store the FAISS index data')
    parser.add_argument('--percentages_for_index_splitting', nargs='*', type=float, default=[],
                        help='Percentages representing the points at which to split the embeddings (used later in case of GPU indexing). If you want to split the embeddings into n portions, specify only n - 1 percentages. Each value is a percentage of the total corpus size, e.g., 40 means 40%% of the corpus. If no percentage is provided, only one index is created with all the embeddings.')
    parser.add_argument('--output_dir', type=str, default='data/corpus/embeddings/', help='Output directory of the saved embeddings')
    parser.add_argument('--prefix_name', type=str, default='contriever', help='Initial part of the name of the saved embeddings')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size used in `compute_corpus_embeddings` for embedding documents')
    parser.add_argument('--save_every', type=int, default=500, help='Save every steps used in `compute_corpus_embeddings`')
    
    args = parser.parse_args()

    if any(p <= 0 for p in args.percentages_for_index_splitting):
        raise ValueError("Percentages for index splitting must be positive.")

    if sum(args.percentages_for_index_splitting) >= 100:
        raise ValueError("The sum of the percentages for index splitting must be less than 100. The remaining percentage is automatically computed.")

    return args


def load_all_embeddings(args: argparse.Namespace) -> np.array:
    all_embeddings_path = f'{args.output_dir}/{args.prefix_name}_all_embeddings.npy'

    # Check if the file with all embeddings already exists and in case load it
    if os.path.isfile(all_embeddings_path):
        embeddings = np.load(all_embeddings_path, mmap_mode='c')
        return embeddings


    all_embeddings = []
    num_embed = args.batch_size * args.save_every

    for i in range(num_embed - 1, args.corpus_size, num_embed):
        emb_path = f'{args.output_dir}/{args.prefix_name}_{i}_embeddings.npy'
        emb = np.load(emb_path, mmap_mode='c')
        all_embeddings.append(emb)

    last_idx = args.corpus_size - 1
    last_emb_path = f'{args.output_dir}/{args.prefix_name}_{last_idx}_embeddings.npy'
    last_emb = np.load(last_emb_path, mmap_mode='c')
    all_embeddings.append(last_emb)

    embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(all_embeddings_path, embeddings)

    return embeddings


def indexing_embeddings(args: argparse.Namespace, embeddings: np.array) -> None:
    os.makedirs(args.faiss_dir, exist_ok=True)

    # No splitting, hence use only one index
    if not args.percentages_for_index_splitting:
        index = Indexer(args.vector_sz, idx_type=args.idx_type)
        index.index_data(list(range(args.corpus_size)), embeddings)

        print(f"Saving index...")
        index.serialize(
            dir_path=args.faiss_dir, 
            index_file_name=f'{args.idx_type}_index.faiss', 
            meta_file_name=f'{args.idx_type}_index_meta.faiss'
        )
        print(f"Index saved")
        return 
    
    # Split embeddings
    for i in range(len(args.percentages_for_index_splitting)):
        start_idx = int((args.corpus_size * sum(args.percentages_for_index_splitting[:i])) / 100)
        end_idx = int((args.corpus_size * sum(args.percentages_for_index_splitting[:i+1])) / 100)

        print(f"Splitting {i + 1} with documents from {start_idx} to {end_idx} excluded")

        index = Indexer(args.vector_sz, idx_type=args.idx_type)
        index.index_data(list(range(start_idx, end_idx)), embeddings[start_idx: end_idx])

        print(f"Saving index {i + 1}...")
        index.serialize(
            dir_path=args.faiss_dir, 
            index_file_name=f'{args.idx_type}_index{i + 1}.faiss', 
            meta_file_name=f'{args.idx_type}_index{i + 1}_meta.faiss'
        )
        print(f"Index {i + 1} saved")

    # Last split
    print(f"Splitting {i + 2} with documents from {end_idx} to {args.corpus_size} excluded")
    index = Indexer(args.vector_sz, idx_type=args.idx_type)
    index.index_data(list(range(end_idx, args.corpus_size)), embeddings[end_idx:])
    
    print(f"Saving index {i + 2}...")
    index.serialize(
        dir_path=args.faiss_dir, 
        index_file_name=f'{args.idx_type}_index{i + 2}.faiss', 
        meta_file_name=f'{args.idx_type}_index{i + 2}_meta.faiss'
    )
    print(f"Index {i + 2} saved")


def main():
    args = parse_arguments()

    print("Loading embeddings...")
    embeddings = load_all_embeddings(args)
    print("Embeddings loaded")

    print("Indexing embeddings...")
    indexing_embeddings(args, embeddings)
    print("Indexing done")

if __name__ == '__main__':
    seed_everything(SEED)
    main()