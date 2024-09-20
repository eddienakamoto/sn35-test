import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoConfig, AdamW, BitsAndBytesConfig
from sklearn.model_selection import train_test_split
import argparse
import logging
from tqdm import tqdm
import random
from accelerate import Accelerator
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import bitsandbytes as bnb

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(file_path, test_mode=False):
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)

    if test_mode:
        logger.info("Running in test mode with 10 records")
        df = df.sample(n=10, random_state=42)

    formatted_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Formatting data"):
        # Ensure all fields are strings and handle missing values
        input_text = str(row['logic_question']) if pd.notnull(
            row['logic_question']) else ""
        reasoning = str(row['logic_reasoning']) if pd.notnull(
            row['logic_reasoning']) else ""
        answer = str(row['logic_answer']) if pd.notnull(
            row['logic_answer']) else ""
        formatted_data.append({
            "input": input_text,
            "output": f"Reasoning: {reasoning}\n\nGive me the final short answer as a sentence. Don't reason anymore, just say the final answer in math latex.\n\nAnswer: {answer}"
        })

    return formatted_data


def tokenize_function(examples, tokenizer, max_length):
    model_inputs = tokenizer(
        examples["input"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    labels = tokenizer(
        examples["output"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        # 'text_target=True'
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load model with 8-bit quantization
    logger.info(f"Loading model: {args.model_name}")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=args.load_in_8bit) if args.load_in_8bit else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Use bitsandbytes AdamW optimizer
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.learning_rate)

    # Add Accelerator
    accelerator = Accelerator()

    # Move model to device after Accelerator initialization
    model = accelerator.prepare(model)

    # Prepare dataset
    data = load_data(args.data_file, test_mode=args.test_mode)
    train_data, test_data = train_test_split(
        data, test_size=args.test_size, random_state=42)

    logger.info(f"Train set size: {len(train_data)
                                   }, Test set size: {len(test_data)}")

    train_dataset = Dataset.from_dict({"input": [item["input"] for item in train_data], "output": [
                                      item["output"] for item in train_data]})
    test_dataset = Dataset.from_dict({"input": [item["input"] for item in test_data], "output": [
                                     item["output"] for item in test_data]})

    logger.info("Tokenizing datasets")
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(
            examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_test_dataset = test_dataset.map(
        lambda examples: tokenize_function(
            examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=test_dataset.column_names
    )

    # Create a custom data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="max_length",
        max_length=args.max_length,
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_grad_norm=0.3,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=10 if args.test_mode else -1,  # Limit steps in test mode
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=True,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),  # (optimizer, scheduler)
    )

    # Use Accelerator for training
    with accelerator.main_process_first():
        trainer.train()

    # Save the model with LoRA adapters
    logger.info("Saving the fine-tuned model with LoRA adapters...")
    model.save_pretrained(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)
    logger.info("Fine-tuning complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model on LogicNet data")
    parser.add_argument("--data_file", type=str,
                        default="processed_data.csv", help="Path to the CSV data file")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct",
                        help="Name or path of the pre-trained model")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--model_save_path", type=str,
                        default="./fine_tuned_model", help="Path to save the final model")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float,
                        default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int,
                        default=500, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float,
                        default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int,
                        default=10, help="Log every X steps")
    parser.add_argument("--eval_steps", type=int,
                        default=500, help="Evaluate every X steps")
    parser.add_argument("--save_steps", type=int, default=10000,
                        help="Save checkpoint every X steps")
    parser.add_argument("--save_total_limit", type=int,
                        default=2, help="Limit the total amount of checkpoints")
    parser.add_argument("--test_size", type=float, default=0.1,
                        help="Proportion of the dataset to include in the test split")
    parser.add_argument("--logging_dir", type=str,
                        default="./logs", help="Directory for storing logs")
    parser.add_argument("--test_mode", action="store_true",
                        help="Run in test mode with 10 records")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--max_length", type=int,
                        default=2048, help="Maximum sequence length")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 mixed precision training")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit quantization")
    args = parser.parse_args()
    main(args)
