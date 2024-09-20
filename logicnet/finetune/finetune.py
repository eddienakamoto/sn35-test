from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
import json

# 1. Load the Dataset
# Make sure the file is structured properly, as shown above
dataset = load_dataset(
    'json',
    data_files={
        'train': 'data/output.json',
        'eval': 'data/output.json'}
)

# Load the training and evaluation datasets
train_dataset = dataset['train']
eval_dataset = dataset['eval']

# 2. Load the Qwen2 Model and Tokenizer
model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True)

# 3. Add Special Tokens if Necessary (optional)
# If your dataset uses special tokens (e.g., '<|user|>' or '<|assistant|>'), add them here
special_tokens_dict = {
    'additional_special_tokens': ['<|user|>', '<|assistant|>']}
tokenizer.add_special_tokens(special_tokens_dict)
# Resize the embeddings to match the new vocab size
model.resize_token_embeddings(len(tokenizer))

# 4. Define a Data Collator
# Since this is a Causal Language Model, MLM (Masked Language Modeling) should be False
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 5. Preprocess the Data
# We need to tokenize the conversation history and the expected response


def preprocess_function(examples):
    # Initialize input text
    inputs = ""

    # Check and process the conversation field properly
    for conversation in examples["conversation"]:
        # Ensure each entry in the conversation is a dictionary
        if isinstance(conversation, dict):
            if conversation["role"] == "user":
                inputs += "<|user|> " + conversation["content"] + " "
            elif conversation["role"] == "assistant":
                inputs += "<|assistant|> " + conversation["content"] + " "
        else:
            print(f"Unexpected conversation format: {conversation}")

    # Tokenize the input and the expected response
    model_inputs = tokenizer(inputs, truncation=True,
                             max_length=702, padding="max_length")

    # Tokenize the expected response (label)
    labels = tokenizer(
        examples['response'], truncation=True, max_length=702, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# Apply the preprocessing to the entire dataset without using cache
train_dataset = train_dataset.map(
    preprocess_function, batched=True, load_from_cache_file=False)
eval_dataset = eval_dataset.map(
    preprocess_function, batched=True, load_from_cache_file=False)

# 6. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./qwen2-finetuned",  # Directory to save the model
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    per_device_train_batch_size=1,  # Adjust based on your hardware
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Adjust to manage memory with smaller batch sizes
    num_train_epochs=3,  # Number of epochs for fine-tuning
    fp16=True,  # Enable mixed precision training for faster training and less memory usage
    learning_rate=5e-5,  # Adjust the learning rate as needed
    weight_decay=0.01,
    push_to_hub=False  # Set to True if you want to push to Hugging Face Hub
)

# 7. Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 8. Train the Model
trainer.train()

# 9. Save the Fine-Tuned Model
model.save_pretrained("./qwen2-finetuned")
tokenizer.save_pretrained("./qwen2-finetuned")
