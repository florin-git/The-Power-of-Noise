import os
from typing import List, Optional, Dict
from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoTokenizer

import numpy as np
import torch
import torch.nn.functional as F

import normalize_text


class Encoder(PreTrainedModel):
    """
    A wrapper class for encoding text using pre-trained transformer models with specified pooling strategy.
    """
    def __init__(self, config: AutoConfig, pooling: str = "average"):
        super().__init__(config)
        self.config = config
        if not hasattr(self.config, "pooling"):
            self.config.pooling = pooling

        self.model = AutoModel.from_pretrained(
            config.name_or_path, config=self.config
        )


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    
    def encode(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        normalize: bool = False
    ) -> torch.Tensor:
        model_output = self.forward(
            input_ids, 
            attention_mask,
            token_type_ids,
        )
        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = F.normalize(emb, dim=-1)

        return emb


class Retriever:
    """
    A class for retrieving document embeddings using a specified encoder, using a bi-encoder approach.
    """
    def __init__(
        self,
        device: torch.device,
        tokenizer: AutoTokenizer,
        query_encoder: Encoder,
        doc_encoder: Optional[Encoder] = None,
        max_length: int = 512,
        add_special_tokens: bool = True,
        norm_query_emb: bool = False,
        norm_doc_emb: bool = False,
        lower_case: bool = False,
        do_normalize_text: bool = False,
    ):
        
        self.device = device
        self.query_encoder = query_encoder.to(device)
        self.doc_encoder = self.query_encoder if doc_encoder is None else doc_encoder.to(device)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query_emb = norm_query_emb
        self.norm_doc_emb = norm_doc_emb
        self.lower_case = lower_case
        self.do_normalize_text = do_normalize_text


    def encode_queries(self, queries: List[str], batch_size: int) -> np.ndarray:
        if self.do_normalize_text:
            queries = [normalize_text.normalize(q) for q in queries]
        if self.lower_case:
            queries = [q.lower() for q in queries]

        all_embeddings = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))

                q_inputs = self.tokenizer(
                    queries[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                ).to(self.device)

                emb = self.query_encoder.encode(**q_inputs, normalize=self.norm_query_emb)
                all_embeddings.append(emb.cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        return all_embeddings
    

    def encode_corpus(
        self, 
        corpus_info: List[Dict[str, str]], 
        batch_size: int, 
        output_dir: str, 
        prefix_name: str,
        save_every: int = 500
    ) -> None:
        all_embeddings = []
        num_steps = 0

        nbatch = (len(corpus_info) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(corpus_info))

                corpus = [
                    c["title"] + " " + c["text"] if len(c["title"]) > 0 else c["text"] 
                    for c in corpus_info[start_idx: end_idx]
                ]
                if self.do_normalize_text:
                    corpus = [normalize_text.normalize(c) for c in corpus]
                if self.lower_case:
                    corpus = [c.lower() for c in corpus]

                doc_inputs = self.tokenizer(
                    corpus,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                ).to(self.device)

                emb = self.doc_encoder.encode(**doc_inputs, normalize=self.norm_doc_emb)
                all_embeddings.append(emb)

                num_steps += 1

                if num_steps == save_every or k == nbatch - 1:
                    embeddings = torch.cat(all_embeddings, dim=0)
                    file_index = end_idx - 1  # Index of the last passage embedded in the batch
                    file_path = os.path.join(
                        output_dir, f'{prefix_name}_{file_index}_embeddings.npy'
                    )
                    np.save(file_path, embeddings.cpu().numpy())
                    print(f"Saved embeddings for {file_index} passages.")
                    num_steps = 0
                    all_embeddings = []