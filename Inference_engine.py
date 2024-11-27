import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, LoraConfig, get_peft_model

# Define device (Use GPU if available, else fallback to CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define label maps
id2label = {0: "UNDEFINED", 1: "LEFT", 2: "RIGHT", 3: "CENTER"}
label2id = {"UNDEFINED": 0, "LEFT": 1, "RIGHT": 2, "CENTER": 3}

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", add_prefix=True)

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

# Load LoRA-enhanced model weights
model.load_state_dict(
    torch.load("/Users/ilseoplee/NLPizza_final_project/Filing/LORA_Body/trained_model_gral_imbd_body_2017_1_shawn.pth", map_location=device)
)
model.to(device)
model.eval()  # Set model to evaluation mode

# Load CSV file
input_file = "2017_2.csv"  # Replace with your input file path
df = pd.read_csv(input_file)

# Ensure the "body" column exists in the input file
if "body" not in df.columns or "political_leaning" not in df.columns:
    raise ValueError("The input file must contain 'body' and 'political_leaning' columns.")

# Preprocess and tokenize data
def preprocess_and_tokenize(text):
    """
    Tokenizes input text for the model.
    Returns tokenized inputs in the required format.
    """
    if not isinstance(text, str) or text.strip() == "":
        return None  # Ignore non-string or empty text
    try:
        # Tokenize input
        return tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return None

# Perform inference
print("Running inference on the dataset...")
inference_results = []
with torch.no_grad():  # Disable gradient calculation for inference
    for text in tqdm(df["body"], desc="Processing rows"):
        tokenized_inputs = preprocess_and_tokenize(text)
        if tokenized_inputs is None:
            inference_results.append("N/A")  # Handle invalid text
            continue

        # Forward pass
        logits = model(**tokenized_inputs).logits

        # Get predicted class
        prediction = torch.argmax(logits, dim=-1).item()
        inference_results.append(id2label[prediction])

# Add "inference" column to the DataFrame
df["inference"] = inference_results

# Create "test_result" column by comparing "inference" with "political_leaning"
df["test_result"] = df.apply(
    lambda row: "N/A" if row["inference"] == "N/A" else ("correct" if row["inference"] == row["political_leaning"] else "wrong"),
    axis=1,
)

# Save the updated DataFrame to a new CSV file
output_file = "2017_2_test.csv"  # Replace with your desired output file path
df.to_csv(output_file, index=False)

print(f"Results saved to {output_file}")
