import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from datasets import load_dataset

dataset = load_dataset(
    'json',
    data_files={
        'train': 'data/output.json',
        'eval': 'data/output.json'}
)

# Load the training and evaluation datasets
train_dataset = dataset['train']
eval_dataset = dataset['eval']

# Load the tokenizer
model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Function to calculate token lengths without truncation


def compute_token_lengths(examples):
    # Concatenate the conversation into a single string
    conversations = examples["conversation"]
    inputs = ""
    for message in conversations:
        if message["role"] == "user":
            inputs += "<|user|> " + message["content"] + " "
        elif message["role"] == "assistant":
            inputs += "<|assistant|> " + message["content"] + " "

    # Tokenize without truncation
    tokens = tokenizer(inputs, truncation=False)
    length = len(tokens["input_ids"])  # Get the token length
    return {"length": length}


# Apply this function to the dataset
lengths_dataset = train_dataset.map(compute_token_lengths, batched=False)

# Extract the lengths for analysis
lengths = lengths_dataset["length"]

# Plot the distribution of token lengths
plt.hist(lengths, bins=50)
plt.title("Distribution of Token Lengths")
plt.xlabel("Token Length")
plt.ylabel("Frequency")
plt.show()

# Get statistics for the lengths
print(f"Max Length: {np.max(lengths)}")
print(f"99th Percentile: {np.percentile(lengths, 99)}")
print(f"95th Percentile: {np.percentile(lengths, 95)}")
print(f"90th Percentile: {np.percentile(lengths, 90)}")
