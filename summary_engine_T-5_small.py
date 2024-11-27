from datasets import Dataset
from transformers import pipeline
import pandas as pd
import os
import torch

# Check if MPS is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load dataset
input_file = "2017_1_1.csv"
df = pd.read_csv(input_file)

# Add 'summary' column if not exists
if "summary" not in df.columns:
    df["summary"] = ""

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Summarization pipeline with MPS support
summarizer = pipeline(
    "summarization",
    model="t5-small",
    tokenizer="t5-small",
    device=0 if device.type == "mps" else -1,  # 0: GPU(MPS), -1: CPU
    clean_up_tokenization_spaces=False,
    max_length=100,
    batch_size=64
)

# Summarization process
def summarize_batch(batch):
    summaries = []
    for text in batch["body"]:
        try:
            result = summarizer(
                text,
                max_length=100,
                min_length=10,
                truncation=True
            )
            summaries.append(result[0]["summary_text"])
        except Exception as e:
            print(f"Error summarizing: {e}")
            summaries.append("")
    batch["summary"] = summaries
    return batch

# Save progress
def save_progress(batch, output_file, batch_index):
    temp_file = f"{os.path.splitext(output_file)[0]}_batch_{batch_index}.csv"
    batch.to_csv(temp_file, index=False, encoding="utf-8-sig")
    print(f"Progress saved at batch {batch_index} to {temp_file}")

# Summarization process
def summarize_data(dataset, output_file, batch_size=64):
    total_batches = len(dataset) // batch_size + 1
    for i in range(total_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(dataset))
        print(f"Processing batch {i + 1}/{total_batches} ({start}-{end})...")
        batch = dataset.select(range(start, end)).to_pandas()
        summarized_batch = summarize_batch(batch)
        save_progress(summarized_batch, output_file, i + 1)

    print(f"Summarization completed. Results saved to {output_file}")

# Output file
output_file_path = "summarized_data_2017_1_1.csv"

# Run summarization
summarize_data(dataset, output_file_path, batch_size=64)


# # Colab

# from datasets import Dataset
# from transformers import pipeline
# from accelerate import Accelerator
# import pandas as pd
# import os

# # Load dataset
# input_file = "2017_1_1.csv"
# df = pd.read_csv(input_file)

# # Add 'summary' column if not exists
# if "summary" not in df.columns:
#     df["summary"] = ""

# # Convert to Hugging Face Dataset
# dataset = Dataset.from_pandas(df)

# # Initialize Accelerate
# accelerator = Accelerator()
# device = accelerator.device
# print(f"Using device: {device}")

# # Summarization pipeline with clean_up_tokenization_spaces and max_length
# summarizer = pipeline(
#     "summarization",
#     model="t5-small",
#     tokenizer="t5-small",
#     device_map="auto",
#     batch_size=400,
#     clean_up_tokenization_spaces=False,  # Avoids the warning
#     max_length=100,  # Ensures truncation happens
# )


# # Function to summarize a batch
# def summarize_batch(batch):
#     summaries = []
#     for text in batch["body"]:
#         try:
#             result = summarizer(
#                 text,
#                 max_length=100,
#                 min_length=10,
#                 truncation=True,
#                 clean_up_tokenization_spaces=False,  # Avoids warning
#             )
#             summaries.append(result[0]["summary_text"])
#         except Exception as e:
#             print(f"Error summarizing: {e}")
#             summaries.append("")
#     batch["summary"] = summaries
#     return batch


# # Save progress
# def save_progress(batch, output_file, batch_index):
#     temp_file = f"{os.path.splitext(output_file)[0]}_batch_{batch_index}.csv"
#     batch.to_csv(temp_file, index=False, encoding="utf-8-sig")
#     print(f"Progress saved at batch {batch_index} to {temp_file}")


# # Summarization process
# def summarize_data(dataset, output_file, batch_size=64):
#     total_batches = len(dataset) // batch_size + 1
#     for i in range(total_batches):
#         start = i * batch_size
#         end = min((i + 1) * batch_size, len(dataset))
#         print(f"Processing batch {i + 1}/{total_batches} ({start}-{end})...")
#         batch = dataset.select(range(start, end)).to_pandas()
#         summarized_batch = summarize_batch(batch)
#         save_progress(summarized_batch, output_file, i + 1)

#     print(f"Summarization completed. Results saved to {output_file}")


# # Output file
# output_file_path = "summarized_data_2017_1_1.csv"

# # Run summarization
# summarize_data(dataset, output_file_path, batch_size=64)
