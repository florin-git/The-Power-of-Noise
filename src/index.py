import os 
import faiss
import numpy as np
from typing import List, Tuple, Optional

from utils import read_pickle, write_pickle


# Indexer class adapted from Contriever file https://github.com/facebookresearch/contriever/blob/main/src/index.py

class Indexer(object):
    """
    Initializes an indexer with a specified vector size and index type, optionally placing the index on a GPU.

    Attributes:
        vector_sz (int): The size of the vectors to be indexed.
        idx_type (str): The type of index ('IP' for Inner Product or 'L2' for Euclidean distance).
        gpu_id (Optional[int]): Optional GPU ID for GPU acceleration.
        index_id_to_db_id (List[int]): A list of external IDs for the indexed vectors.
    """
    def __init__(self, vector_sz: int, idx_type: str = 'IP', gpu_id: Optional[int] = None):
        self.idx_type = idx_type
        self.gpu_id = gpu_id

        if idx_type == 'IP':
            quantizer = faiss.IndexFlatIP(vector_sz)
        elif idx_type == 'L2':
            quantizer = faiss.IndexFlatL2(vector_sz)
        else:
            raise NotImplementedError('Only L2 norm and Inner Product metrics are supported')
        
        self.index = quantizer
        self.index_id_to_db_id = []


    def index_data(self, ids: List[int], embeddings: np.array):
        """
        Adds data to the index.

        Args:
            ids (List[int]): A list of database IDs corresponding to the embeddings.
            embeddings (np.array): A numpy array of embeddings to be indexed.
        """
        self._update_id_mapping(ids)
        # embeddings = embeddings.astype('float32')
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)

        print(f'Total data indexed {len(self.index_id_to_db_id)}')

    def search_knn(
        self, 
        query_vectors: np.array, 
        top_docs: int, 
        index_batch_size: int = 2048
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Performs a k-nearest neighbor search for the given query vectors.

        Args:
            query_vectors (np.array): A numpy array of query vectors.
            top_docs (int): The number of top documents to return for each query.
            index_batch_size (int): The batch size to use when indexing.
        
        Returns:
            A list of tuples, each containing a list of document IDs and a list of corresponding scores.
        """
        result = []
        nbatch = (len(query_vectors)-1) // index_batch_size + 1
        for k in range(nbatch):
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def serialize(
        self, 
        dir_path: str, 
        index_file_name: Optional[str] = None, 
        meta_file_name: Optional[str] = None
    ):
        """
        Serializes the index and its metadata to disk.

        Args:
            dir_path (str): The directory path to save the serialized index and metadata.
            index_file_name (Optional[str]): Optional custom name for the index file.
            meta_file_name (Optional[str]): Optional custom name for the metadata file.
        """
        if index_file_name is None:
            index_file_name = f'{self.idx_type}_index.faiss'
        if meta_file_name is None:
            meta_file_name = f'{self.idx_type}_index_meta.faiss'

        index_file = os.path.join(dir_path, index_file_name)
        meta_file = os.path.join(dir_path, meta_file_name)
        print(f'Serializing index to {index_file}, meta data to {meta_file}')

        faiss.write_index(self.index, index_file)
        write_pickle(self.index_id_to_db_id, meta_file)


    def deserialize_from(
        self, 
        dir_path: str, 
        index_file_name: Optional[str] = None, 
        meta_file_name: Optional[str] = None
    ):
        """
        Loads the index and its metadata from disk.

        Args:
            dir_path (str): The directory path from where to load the index and metadata.
            index_file_name (Optional[str]): Optional custom name for the index file.
            meta_file_name (Optional[str]): Optional custom name for the metadata file.
        """
        if index_file_name is None:
            index_file_name = f'{self.idx_type}_index.faiss'
        if meta_file_name is None:
            meta_file_name = f'{self.idx_type}_index_meta.faiss'

        index_file = os.path.join(dir_path, index_file_name)
        meta_file = os.path.join(dir_path, meta_file_name)
        print(f'Loading index from {index_file}, meta data from {meta_file}')

        self.index = faiss.read_index(index_file)
        print('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)

        self.index_id_to_db_id = read_pickle(meta_file)
        assert len(
            self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'
        
        # Move index to GPU if specified
        if self.gpu_id is not None:
            res = faiss.StandardGpuResources()  
            self.index_gpu = faiss.index_cpu_to_gpu(res, self.gpu_id , self.index)
            del self.index
            self.index = self.index_gpu
            print('Moved index to GPU %d', self.gpu_id)
        

    def _update_id_mapping(self, db_ids: List[int]):
        self.index_id_to_db_id.extend(db_ids)

    def get_index_name(self):
        return f"{self.idx_type}_index"
    

def merge_ip_search_results(
    indexer1: Indexer, 
    indexer2: Indexer, 
    query_vectors: np.array, 
    top_docs: int, 
    index_batch_size: int = 2048
) -> List[Tuple[List[str], List[float]]]:
    """
    Merges the k-nearest neighbor search results from two different indices for a given set of query vectors.

    Args:
        indexer1 (Indexer): The first indexer object capable of performing knn searches.
        indexer2 (Indexer): The second indexer object capable of performing knn searches.
        query_vectors (np.array): A numpy array of query vectors for which to perform the searches.
        top_docs (int): The number of top documents to retrieve from the combined results of the two indexer.
        index_batch_size (int): The batch size to use for indexing operations.
    
    Returns:
        A list of tuples, where each tuple contains two lists - the merged list of database IDs and the corresponding scores.
    """
    # Perform searches on both indices
    results1 = indexer1.search_knn(query_vectors, top_docs, index_batch_size)
    results2 = indexer2.search_knn(query_vectors, top_docs, index_batch_size)

    merged_results = []
    for res1, res2 in zip(results1, results2):
        # Merge the results from both indices
        combined_db_ids = res1[0] + res2[0]
        combined_scores = res1[1] + res2[1]

        # Since we're using inner product, higher scores indicate better matches
        # Combine and sort the results by score in descending order
        combined = sorted(zip(combined_db_ids, combined_scores), key=lambda x: x[1], reverse=True)

        # Get only the top_docs results after merging
        combined = combined[:top_docs]

        # Separate the db_ids and scores again
        db_ids, scores = zip(*combined)

        merged_results.append((list(db_ids), list(scores)))

    return merged_results