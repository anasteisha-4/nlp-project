# test_reranker.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "C:/Users/Alex/Desktop/nlp-project/models/reranker"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=1
)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
examples = [
    {
        "query": "Jak zrobić ciasto czekoladowe?",
        "passage": "Ciasto czekoladowe można przygotować z mąki, cukru, jajek i gorzkiej czekolady. Upiec w 180°C przez 30 minut."
    },
    {
        "query": "Jak zrobić ciasto czekoladowe?",
        "passage": "Samochód elektryczny Tesla Model 3 ma zasięg do 600 km na jednym ładowaniu."
    },
    {
        "query": "Gdzie znajduje się Kraków?",
        "passage": "Kraków to miasto w południowej Polsce, leżące nad Wisłą. Jest jednym z najstarszych miast w kraju."
    },
    {
        "query": "Gdzie znajduje się Kraków?",
        "passage": "Paryż to stolica Francji, znana z Wieży Eiffla i Luwru."
    }
]

for i, ex in enumerate(examples, 1):
    inputs = tokenizer(
        ex["query"],
        ex["passage"],
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits.item()

    print(i)
    print(f"  query: {ex['query']}")
    print(f"  passage: {ex['passage'][:80]}...")
    print(f"  Score: {score:.4f}\n")