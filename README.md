# The Power of Noise: Redefining Retrieval for RAG Systems

This repository contains the code and data to reproduce the experiments from the paper [The Power of Noise: Redefining Retrieval for RAG Systems](https://arxiv.org/abs/2401.14887).

## Installation

1. Set up a conda environment.

```
conda create -n power_of_noise python=3.9 --yes
conda activate power_of_noise
```

2. Install package and requirements.

```
pip install -r requirements.txt
```

## Data
The corpus and NQ datasets can be downloaded from HuggingFace using the code in the respective sections.

The full training set was not used for the experiments; instead, a smaller sample of 10K entries was employed, and is available in the `data` folder of this repository. For the experiments described in the "RAG in Practice" section, the test set was utilized.

Data not present in this repository or not downloadable from HuggingFace can be found in this [Google Drive](https://drive.google.com/drive/folders/1MfR7mJ76tyVpjbMwUkMVbEjOQKpdL-Lq?usp=sharing).


### Corpus

- **Original Source**: The corpus originates from the English Wikipedia (Dec. 20, 2018), where each document is segmented into non-overlapping passages of 100 words. The original corpus can be downloaded from this [link](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz).

- **Processing**: We integrated gold documents from the [NQ dataset](https://ai.google.com/research/NaturalQuestions) in the corpus, applying duplicate filtering for precise query-document matching. Documents exceeding 512 tokens (tokenized with Llama2) were excluded to maintain manageable input sizes for LLMs, resulting in a final corpus of 21,035,236 documents.

The processed dataset is available on HuggingFace:
```
from datasets import load_dataset
corpus = load_dataset('florin-hf/wiki_dump2018_nq_open')
```


An example of a Wikipedia passage is as follows:
```
{
  "text": Home computers were a class of microcomputers entering the market in 1977, and becoming common during the 1980s.
          They were marketed to consumers as affordable and accessible computers that, for the first time, were intended for the use of a single nontechnical user.
          These computers were a distinct market segment that typically cost much less than business,
          scientific or engineering-oriented computers of the time such as the IBM PC, and were generally less powerful in terms of memory and expandability.
          However, a home computer often had better graphics and sound than contemporary business computers. Their most common uses were playing
  "title": "Home computer"
}
```

#### Subsets of the Corpus 
Considering the substantial memory requirements (~25Gb) for loading the entire corpus, we provide subsets tailored to specific experiments, reducing the RAM footprint.

A subset contains only the documents present in the search results by the retrievers or by random sampling for a the specific configuration (see **Generation** section). In this way, when running the generation, it is not needed to load the entire corpus in RAM, but only the documents that could possibly be included in the prompt of the LLMs. 

These subsets can be found in the Google Drive under the folder `data/processed`. 


### Natural Questions

The NQ dataset, curated to exclude entries with gold documents over 512 tokens, includes 72,209 training and 2,889 test examples. In these experiments, the validation set was not used, but can be downloaded along with the other splits from HuggingFace:
```
from datasets import load_dataset
dataset = load_dataset('florin-hf/nq_open_gold')
```

A sample in the dataset has the following format:
```
{
    'example_id' (int64): an identifier for the question, consistent with the original NQ dataset,
    'question' (str): a question, that is identical to the question in the original NQ,
    'answers' (List[str]): the list of correct answers in the original NQ,
    'text' (str): gold document, associated with the question, in the original NQ,
    'idx_gold_in_corpus' (int64): index of the gold document in the full corpus.
}

Ex.
{
    'example_id': -3440030035760311385,
    'question': 'who owned the millennium falcon before han solo',
    'answers': [Lando Calrissian],
    'text': "Han Solo won the Millennium Falcon from Lando Calrissian in the card game ' sabacc ' several years before the events of the film A New Hope..."
    'idx_gold_in_corpus': 20995349
}
```

## RAG Steps

### 1. Retrieval

In the first phase of a RAG system, a retriever is employed to search the top-ranked documents based on a given similarity metric. In these experiments three different retrievers were used: Contriever, ADORE and BM25. The search results of the three models are presented in the data folder (e.g., `data/contriever_search_results_at150.pkl`). Each result is a tuple containing in the first position the indices of the top-ranked documents in the corpus; and as second position their corresponding scores. In the case of dense retriever, an Inner Product (IP) search was adopted, thus the higher the score the closer the embeddings in the vector space.

The following three steps were used to compute search results for a dense retriever:
##### 1. Compute Corpus Embeddings
The `compute_corpus_embeddings.py` script computes embeddings in batches, storing them in the `output_dir`.

##### 2. Index Embeddings:
The `index_embeddings.py` script concatenates the piecewise stored embeddings and creates the index.
- Single Index: Leave `percentages_for_index_splitting` empty.
- Multiple Indices: Specify splitting percentages to create multiple indices that can be loaded into different GPUs.

##### 3. Retrieve Top-k Documents
The `compute_search_results.py` script retrieves the top-k documents for the given queries using the FAISS index/indices created earlier.

### 2. Generation

Our experiments tested different prompt structures across four LLMs. The table below summarizes the LLMs used:
| LLM Model | HF Link |
|-----------|---------|
| Llama-2-7b-chat-hf | [HF link](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| mpt-7b-instruct  | [HF link](https://huggingface.co/mosaicml/mpt-7b-instruct) |
| phi-2 | [HF link](https://huggingface.co/microsoft/phi-2) |
| falcon-7b-instruct | [HF link](https://huggingface.co/tiiuae/falcon-7b-instruct) |


#### Closed-Book QA

In the closed-book QA configuration, the system generates answers based solely on the question, without external knowledge. The script `src/generate_answers_llm_only_query.py` allows to generate responses using only the task instruction and the query. A corresponding example script can be found in the file `example_scripts/run_generation_only_query.sh`.


#### Gold & Distracting/Random

This configuration aims to replicate the first two tables of the paper, exploring how the inclusion of gold documents and distracting/random documents affects the performance. The script `src/generate_answers_llm.py` is used to manage this setup. For instance, to reproduce a scenario `Far` where the gold document is positioned last in a context of seven documents (one gold and six *distracting*), the script below can be run for Llama-2-7b-chat :
```
python src/generate_answers_llm.py \
    --output_dir data/gen_res \
    --llm_id meta-llama/Llama-2-7b-chat-hf \
    --model_max_length 4096 \
    --load_full_corpus True \
    --use_random False \
    --use_adore False \
    --gold_position 6 \
    --num_documents_in_context 7 \
    --get_documents_without_answer True \
    --batch_size 10 \
    --save_every 250
```

In particular, we set `use_random` and `use_adore` to `False` which will load the default search results, hence the ones from the Contriever. Then, with the parameter `get_documents_without_answer` set to `True` we are specifying to include only the _distracting_ documents, i.e., documents that are assigned a high score by the retriever but do not contain the answer. Finally, we choose the position of the gold (0-indexed) and the total number of documents in the context, respectively with `gold_position` and `num_documents_in_context`.
In case we want to use random documents we set `use_random` to `True`.

#### Retrieved & NQ Random

Focusing on a scenario without gold documents, this setup employs retrieved documents alongside randomly chosen entries from the NQ dataset. The script `src/generate_answers_llm_mixed.py` is designed for this purpose:
```
python src/generate_answers_llm_mixed.py \
    --output_dir data/gen_res \
    --llm_id tiiuae/falcon-7b-instruct \
    --model_max_length 2048 \
    --load_full_corpus False \
    --use_bm25 True \
    --num_retrieved_documents 1 \
    --num_random_documents 2 \
    --put_retrieved_first False \
    --use_test True \
    --batch_size 16 \
    --save_every 250
```

In this case, Falcon-7-instruct is used with the search results from BM25 (`use_bm25` is `True`) on the test set (`use_test` is `True`). The number of retrieved and random documents is regulated by `num_retrieved_documents` and `num_random_documents` respectively. The `put_retrieved_first` to `False` says that the retrieved documents come after the random ones in the context, hence close to the query.

#### Retrieved & Other Random

Expanding on the concept of incorporating randomness into the context, this configuration introduces randomness from varied sources. The `src/generate_answers_llm_multi_corpus.py` script allows experimentation with documents from disparate corpora, such as a [Reddit dataset](https://huggingface.co/datasets/webis/tldr-17) or a collection of nonsensical sentences composed of random words (`data/processed/corpus_with_random_50_words`). 

## 3. Evaluation
To evaluate the LLMs' responses, we use accuracy. In particular, one response is considered correct if at least one of the predefined correct answers is contained within the response produced by the LLM. 

The code for reading and computing the accuracy is present in the `src/read_generation_results.py` file. For each of the generation case there is an example script (e.g., `example_scripts/run_read_gen_res.sh`). 

For example, to read the generation results of the script in the **Gold & Distracting/Random** section, you can run:
```
python example_scripts/run_read_gen_res.py \
    --output_dir data/gen_res \
    --llm_id meta-llama/Llama-2-7b-chat-hf \
    --use_test False \
    --prompt_type classic \
    --use_random False \
    --use_adore False \
    --gold_position 6 \
    --num_documents_in_context 7 \
    --get_documents_without_answer True \
```





## References
If you find this repository useful, please consider giving it a star and citing this work:
```
@inproceedings{Cuconasu_2024, series={SIGIR 2024},
   title={The Power of Noise: Redefining Retrieval for RAG Systems},
   url={http://dx.doi.org/10.1145/3626772.3657834},
   DOI={10.1145/3626772.3657834},
   booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
   publisher={ACM},
   author={Cuconasu, Florin and Trappolini, Giovanni and Siciliano, Federico and Filice, Simone and Campagnano, Cesare and Maarek, Yoelle and Tonellotto, Nicola and Silvestri, Fabrizio},
   year={2024},
   month=jul, collection={SIGIR 2024}
}
```
