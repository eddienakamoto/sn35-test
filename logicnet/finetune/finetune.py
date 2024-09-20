from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import json

# 1. Load the Dataset
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


def preprocess_function(examples):
    conversation = [
        {"role": "user", "content": examples['logic_question']},
        {"role": "assistant", "content": examples['logic_reasoning']},
        {
            "role": "user",
            "content": "Give me the final short answer as a sentence. Don't reason anymore, just say the final answer in math latex."
        },
        {"role": "assistant", "content": examples['logic_answer']}
    ]

    # Combine the conversation into a single string with role markers
    conversation_str = ""
    for message in conversation:
        conversation_str += f"<|{message['role']}|> {message['content']} "

    # Tokenize the entire conversation
    model_inputs = tokenizer(
        conversation_str, truncation=True, max_length=351, padding="max_length", return_tensors="pt"
    )

    # Tokenize the final assistant response (which we want the model to predict)
    labels = tokenizer(
        examples['logic_answer'], truncation=True, max_length=351, padding="max_length", return_tensors="pt"
    )

    # Assign the labels to the model inputs
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


# Apply the preprocessing to the dataset
train_dataset = train_dataset.map(
    preprocess_function, batched=True, load_from_cache_file=False
)
eval_dataset = eval_dataset.map(
    preprocess_function, batched=True, load_from_cache_file=False
)

# 6. Define Training Arguments
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
