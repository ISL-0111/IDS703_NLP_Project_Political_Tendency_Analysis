import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, LoraConfig, get_peft_model

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define label maps
id2label = {0: "UNDEFINED", 1: "LEFT", 2: "RIGHT", 3: "CENTER"}
label2id = {"UNDEFINED": 0, "LEFT": 1, "RIGHT": 2, "CENTER": 3}

# Load tokenizer (ensure it matches the one used during training)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Initialize base model
model_checkpoint = "distilbert-base-uncased"
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=4, id2label=id2label, label2id=label2id
)

# Initialize LoRA configuration
peft_config = LoraConfig(
    task_type="SEQ_CLS",  # Sequence classification
    r=4,
    lora_alpha=32,
    lora_dropout=0.01,
    target_modules=["q_lin"],  # Match your LoRA target modules
)

# Apply LoRA to the base model
model = get_peft_model(base_model, peft_config)

# Load LoRA-enhanced model weights, PUT pth extension file path here !!
model.load_state_dict(
    torch.load("trained_model_gral_imbd_body_2017_1_shawn.pth", map_location=device)
)
model.to(device)
model.eval()  # Set model to evaluation mode

# Example input data for inference
text_list = [
    "Illegal immigrants should be expelled from the country.",
    "Not a fan, don't recommend.",
    "Better than the first one.",
    "Women have the right to choose and abortion should be allowed.",
]

# Inference
predictions_list = []
with torch.no_grad():  # Disable gradient calculation for inference
    for text in text_list:
        # Tokenize input
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        # Forward pass
        logits = model(**inputs).logits

        # Get predicted class
        prediction = torch.argmax(logits, dim=-1).item()
        predictions_list.append(id2label[prediction])

# Print results
for text, prediction in zip(text_list, predictions_list):
    print(f"Text: {text}\nPrediction: {prediction}\n")
