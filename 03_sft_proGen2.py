import os
import torch
import gc
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False 

BASE_MODEL_ID = "progen2-large"
DATA_PATH = Path("data/data_for_sft.csv")
RESULTS_DIR = Path("results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_sft():
    output_dir = Path("models/sft_lora_new")
    csv_output_path = RESULTS_DIR / "sft_loss_history_lora.csv"
    plot_output_path = RESULTS_DIR / "sft_loss_curve.png"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        trust_remote_code=True
    ).to(DEVICE)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})
        
    tokenizer.padding_side = "right"

    print("Preparing dataset...")
    df = pd.read_csv(DATA_PATH)
    
    def format_text(row):
        return f"{str(row['sequence'])}{tokenizer.eos_token}"

    df['text'] = df.apply(format_text, axis=1)
    dataset = Dataset.from_pandas(df[['text']])
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["qkv_proj", "out_proj", "fc_in", "fc_out"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        num_train_epochs=10, 
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=64,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print("Starting SFT training...")
    trainer.train()
    
    print("Saving model...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Extracting logs and plotting loss curve...")
    log_history = trainer.state.log_history
    train_logs = [entry for entry in log_history if 'loss' in entry and 'step' in entry]
    eval_logs = [entry for entry in log_history if 'eval_loss' in entry and 'step' in entry]
            
    df_train = pd.DataFrame(train_logs)[['step', 'loss']].rename(columns={'loss': 'train_loss'})
    df_eval = pd.DataFrame(eval_logs)[['step', 'eval_loss']]
    
    df_logs = pd.merge(df_train, df_eval, on='step', how='outer').sort_values('step')
    df_logs.to_csv(csv_output_path, index=False)

    plt.figure(figsize=(10, 6))
    if not df_train.empty:
        plt.plot(df_train['step'], df_train['train_loss'], label='Train Loss', color='#7FBCE6', alpha=0.8)
    if not df_eval.empty:
        plt.plot(df_eval['step'], df_eval['eval_loss'], label='Validation Loss', color='#F06055', marker='o')
        
    plt.xlabel('Training Steps', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.title('SFT Training & Validation Loss Curve', fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    print(f"Loss curve plot saved to: {plot_output_path}")

    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    print("SFT pipeline completed successfully!")

if __name__ == "__main__":
    run_sft()