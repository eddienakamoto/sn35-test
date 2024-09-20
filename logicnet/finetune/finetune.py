from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import json

# 1. Define the prompt format with user/assistant roles and EOS token
qwen_prompt = """<|user|> {} <|assistant|> {} <|user|> Please give the final short answer in math latex. <|assistant|> {}"""
EOS_TOKEN = "<|endoftext|>"  # EOS token to mark the end of sequences

# 2. Load the Dataset
dataset = load_dataset(
    'json',
    data_files={
        'train': 'data/output.json',
        'eval': 'data/output.json'}
)

train_dataset = dataset['train']
eval_dataset = dataset['eval']

# 3. Load the Qwen2 Model and Tokenizer
model_name = "Qwen/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True)

# 4. Add Special Tokens if Necessary (optional)
special_tokens_dict = {
    'additional_special_tokens': ['<|user|>', '<|assistant|>']}
tokenizer.add_special_tokens(special_tokens_dict)
# Resize embeddings to match the new vocab size
model.resize_token_embeddings(len(tokenizer))

# 5. Define a Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    # Causal language modeling doesn't require masked language modeling (MLM)
    mlm=False
)

# 6. Preprocess the Data


def preprocess_function(examples):
    # Format the data using the user/assistant prompt format
    logic_question = examples.get('logic_question', '')
    logic_reasoning = examples.get('logic_reasoning', '')
    logic_answer = examples.get('logic_answer', '')

    # Combine the logic_question, logic_reasoning, and logic_answer into the user/assistant prompt
    formatted_prompt = qwen_prompt.format(
        logic_question, logic_reasoning, logic_answer) + EOS_TOKEN

    # Tokenize the entire conversation
    model_inputs = tokenizer(
        formatted_prompt, truncation=True, max_length=1, padding="max_length", return_tensors="pt"
    )

    # Use the same input tokens as labels for causal language modeling
    model_inputs["labels"] = model_inputs["input_ids"].clone()

    return model_inputs


# Apply the preprocessing function to the dataset
train_dataset = train_dataset.map(
    preprocess_function, batched=True, load_from_cache_file=False)
eval_dataset = eval_dataset.map(
    preprocess_function, batched=True, load_from_cache_file=False)

# 7. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./qwen2-finetuned",  # Directory to save the model
    overwrite_output_dir=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    per_device_train_batch_size=2,  # Adjust based on your hardware
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Adjust to manage memory with smaller batch sizes
    num_train_epochs=3,  # Number of epochs for fine-tuning
    fp16=True,  # Enable mixed precision training for faster training and less memory usage
    learning_rate=5e-5,  # Adjust the learning rate as needed
    weight_decay=0.01,
    push_to_hub=False  # Set to True if you want to push to Hugging Face Hub
)

# 8. Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 9. Train the Model
trainer.train()

# 10. Save the Fine-Tuned Model
model.save_pretrained("./qwen2-finetuned")
tokenizer.save_pretrained("./qwen2-finetuned")
