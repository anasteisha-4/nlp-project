import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import CrossEncoder
from tqdm import tqdm

MODEL_NAME = "sdadas/polish-reranker-bge-v2"
TRAIN_FILE = "data/processed/reranker_train_pairs.jsonl"
OUTPUT_DIR = "model/reranker"

BATCH_SIZE = 8
EPOCHS = 1
LR = 1e-7
MARGIN = 1.0
MAX_LENGTH = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PairwiseDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["query"], item["pos"], item["neg"]

def train():
    model = CrossEncoder(MODEL_NAME, num_labels=1, max_length=MAX_LENGTH)

    tokenizer = model.tokenizer
    model.model.to(DEVICE)
    model.model.train()

    dataset = PairwiseDataset(TRAIN_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=LR)
    loss_fct = nn.MarginRankingLoss(margin=MARGIN)


    for epoch in range(EPOCHS):
        total_loss = 0.0

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for queries, pos_docs, neg_docs in loop:
            optimizer.zero_grad()
            pos_inputs = tokenizer( list(queries), list(pos_docs), padding=True, truncation=True, 
                                   max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)

            pos_outputs = model.model(**pos_inputs)
            pos_scores = pos_outputs.logits.squeeze(-1)
            neg_inputs = tokenizer(list(queries), list(neg_docs), padding=True, truncation=True,
                                    max_length=MAX_LENGTH, return_tensors="pt").to(DEVICE)

            neg_outputs = model.model(**neg_inputs)
            neg_scores = neg_outputs.logits.squeeze(-1)

            target = torch.ones_like(pos_scores)
            loss = loss_fct(pos_scores, neg_scores, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
    model.save(OUTPUT_DIR)

if __name__ == "__main__":
    train()
