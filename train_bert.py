import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # <-- FIXED HERE
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm

from config import BERT_MODEL_DIR
from app.nlp.preprocessing import basic_clean
from app.utils.data_utils import ensure_dataset


class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = basic_clean(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def load_data():
    csv_path = ensure_dataset()
    df = pd.read_csv(csv_path)
    df["subject"] = df.get("subject", "")
    df["body"] = df.get("body", "")
    df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    return X_train.tolist(), X_val.tolist(), y_train.tolist(), y_val.tolist()


def main():
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    X_train, X_val, y_train, y_val = load_data()

    train_dataset = EmailDataset(X_train, y_train, tokenizer)
    val_dataset = EmailDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=2e-5)  # <-- FIXED

    num_epochs = 2
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - train"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Train loss: {total_loss / max(1, len(train_loader)):.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - val"):
                labels = batch["labels"].to(device)
                batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        if total > 0:
            print(f"Val accuracy: {correct / total:.4f}")

    BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(BERT_MODEL_DIR)
    tokenizer.save_pretrained(BERT_MODEL_DIR)
    print(f"Saved fine-tuned model to {BERT_MODEL_DIR}")


if __name__ == "__main__":
    main()
