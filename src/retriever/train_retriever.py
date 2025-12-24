import json
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

device = torch.device("cuda:0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    with open("data/processed/retriever_train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    train_examples = []
    for item in tqdm(train_data):
        texts = [item["query"], item["positive_passage"]] + item["negative_passages"]
        train_examples.append(InputExample(texts=texts))

    model_name = "intfloat/multilingual-e5-base"
    model = SentenceTransformer(model_name)

    batch_size = 12
    train_dataloader = DataLoader(
        train_examples,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    train_loss = losses.MultipleNegativesRankingLoss(model)
    epochs = 2

    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": 2e-5},
        output_path="models/retriever",
        show_progress_bar=True,
        use_amp=True
    )

if __name__ == "__main__":
    main()