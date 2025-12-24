---
pipeline_tag: text-ranking
tags:
- transformers
- information-retrieval
language: pl
license: gemma
library_name: sentence-transformers
---

<h1 align="center">polish-reranker-bge-v2</h1>

This is a reranker for Polish based on [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) and further fine-tuned on large dataset of text pairs:
- We utilised [RankNet loss](https://icml.cc/Conferences/2015/wp-content/uploads/2015/06/icml_ranking.pdf) and trained the model on the same data as [sdadas/polish-reranker-roberta-v2](https://huggingface.co/sdadas/polish-reranker-roberta-v2)
- [BAAI/bge-reranker-v2.5-gemma2-lightweight](https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight) was used as the teacher model for distillation
- After the training, we merged the original and fine-tuned weights to create the final checkpoint
- We used a custom implementation of XLM-RoBERTa with support for Flash Attention 2. If you want to use these features, load the model with the arguments `trust_remote_code=True` and `attn_implementation="flash_attention_2"`. This is especially important for this model, since [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) supports long contexts of 8192 tokens. For such input length, the inference can be up to 400% faster with Flash Attention in comparison to the original model. 

In most cases, the use of [sdadas/polish-reranker-roberta-v2](https://huggingface.co/sdadas/polish-reranker-roberta-v2) is preferred to this model as it achieves better results for Polish. The main advantage of this model is its context length, so it may perform better on some datasets with long documents.

## Usage (Huggingface Transformers)

The model can be used with Huggingface Transformers in the following way:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

query = "Jak dożyć 100 lat?"
answers = [
    "Trzeba zdrowo się odżywiać i uprawiać sport.",
    "Trzeba pić alkohol, imprezować i jeździć szybkimi autami.",
    "Gdy trwała kampania politycy zapewniali, że rozprawią się z zakazem niedzielnego handlu."
]

model_name = "sdadas/polish-reranker-bge-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda"
)
texts = [f"{query}</s></s>{answer}" for answer in answers]
tokens = tokenizer(texts, padding="longest", max_length=8192, truncation=True, return_tensors="pt").to("cuda")
output = model(**tokens)
results = output.logits.detach().cpu().float().numpy()
results = np.squeeze(results)
print(results.tolist())
```

## Evaluation Results

The model achieves **NDCG@10** of **64.21** in the Rerankers category of the Polish Information Retrieval Benchmark. See [PIRB Leaderboard](https://huggingface.co/spaces/sdadas/pirb) for detailed results.

## Citation

```bibtex
@article{dadas2024assessing,
  title={Assessing generalization capability of text ranking models in Polish}, 
  author={Sławomir Dadas and Małgorzata Grębowiec},
  year={2024},
  eprint={2402.14318},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```