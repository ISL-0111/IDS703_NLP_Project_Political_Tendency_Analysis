from datasets import Dataset
from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    DataCollatorWithPadding,
)
import torch
import re
import contractions
import numpy as np
from tqdm import tqdm
import preprocessor as p

# Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

import re
import contractions
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd

model_checkpoint = 'distilbert-base-uncased'

#Define label maps
id2label = {0:"UNDEFINED" ,1:"LEFT",2:"RIGHT",3:"CENTER"}
label2id = {"UNDEFINED": 0, "LEFT": 1, "RIGHT": 2, "CENTER": 3}
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.SMILEY)

def get_tokenizer_pretrained_model():
    return DistilBertTokenizer.from_pretrained(model_checkpoint, add_prefix=True)

class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained(model_checkpoint, num_labels=4)

        # Freeze DistilBERT parameters
        for param in self.l1.parameters():
            param.requires_grad = False

        self.dropout = torch.nn.Dropout(0.3)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.fc1 = torch.nn.Linear(768, 1024)  # Input dimension is 768 for BERT
        #self.fc2 = torch.nn.Linear(1024, 512)
        self.classifier = torch.nn.Linear(1024, 4)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, input_ids, attention_mask):
        output = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = self.dropout(pooler)
        pooler = self.fc1(pooler)
        pooler = self.relu(pooler)
        pooler = self.dropout(pooler)
        #pooler = self.fc2(pooler)
        #pooler = self.relu(pooler)
        #pooler = self.dropout(pooler)
        #pooler = self.fc3(pooler)
        #pooler = self.softmax(pooler)
        output = self.classifier(pooler)
        output = self.softmax(output)
        return output
    
def tokenize_function(examples):
    #text = examples["body"]
    text = examples["body"]
    labels = examples["political_leaning"]
    tokenizer = get_tokenizer_pretrained_model()
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,#[preprocess(t) for t in text] ,
        return_tensors = "np",
        padding = True,
        truncation = True,
        max_length = 512
        )

    tokenized_inputs["labels"] = [label2id[label] for label in labels]
    return tokenized_inputs

# Preprocessing function
def preprocess(text):
    """ Preprocess the text to clean it for tokenization """
    def is_english_word(word):
        """Function to filter out non-English words."""
        return bool(re.match(r'^[a-zA-Z]+$', word))

    text = text.lower()  # Convert to lowercase
    text = contractions.fix(text)  # Expand contractions (e.g., "don't" -> "do not")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = p.clean(text)  # Clean text using the clean-text library
    return text

class Triage(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.texts = dataset['body']  # Assuming 'text' column contains the raw text
        self.labels = dataset['political_leaning']
        self.document_id = dataset['id']
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        # Get raw text and label for the current index
        text = self.texts[index]
        self.tokenizer.truncation_side = "left"
        #tokenized_inputs = self.tokenizer(
        tokenized_inputs = self.tokenizer.encode_plus(
            preprocess(text),
            None,
            #return_tensors="pt",
            #padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True
        )

        #encoding = tokenize_function({"text": [text], "labels": [label]}, self.tokenizer, self.max_length)
        #Test
        input_ids = tokenized_inputs['input_ids']  # Remove the batch dimension
        attention_mask = tokenized_inputs['attention_mask']  # Remove the batch dimension

        return {
            #'text' : self.texts[index],
            'document_id' : self.document_id[index],
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label2id[self.labels[index]], dtype=torch.float)
        }

    def __len__(self):
        return len(self.texts)


def test_model(model, data_loader, device):
    model.eval()
    all_document_ids = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # Move batch to GPU/CPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Collect predictions and true labels
            preds = torch.argmax(outputs, dim=1)
            all_document_ids.extend(batch["document_id"])
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")

    print("\nTest Results")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Create a DataFrame for exporting
    results_df = pd.DataFrame({
        "document_id": all_document_ids,
        "true_label": all_labels,
        "predicted_label": all_preds
    })

    return accuracy, precision, recall, f1, results_df