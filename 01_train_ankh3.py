import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datasets import Dataset, ClassLabel
from transformers import (
    T5EncoderModel, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from peft import AdaLoraConfig, get_peft_model, TaskType
from pathlib import Path

MODEL_NAME = "ankh3-large"
DATASET_PATH = Path("data/classification.csv")
OUTPUT_DIR = Path("models/ankh3_adalora_finetune")

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        return -mean_log_prob_pos.mean()

class ContrastiveModel(nn.Module):
    def __init__(self, encoder, hidden_dim=1536, proj_dim=128):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config
        self.supports_gradient_checkpointing = True
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_features = sum_embeddings / sum_mask
        
        projected = self.projection_head(pooled_features)
        return F.normalize(projected, dim=1)

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()

class SupConTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = SupConLoss(temperature=0.07)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        features = model(**inputs)
        loss = self.loss_fct(features, labels)
        return (loss, features) if return_outputs else loss

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(str(DATASET_PATH))
    
    dataset = Dataset.from_pandas(df[['sequence', 'labels']])
    dataset = dataset.cast_column("labels", ClassLabel(num_classes=2))
    dataset = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="labels")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_fn(examples):
        return tokenizer(examples["sequence"], truncation=True, max_length=256)
    
    tokenized_dataset = dataset.map(tokenize_fn, batched=True).remove_columns(["sequence"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_encoder = T5EncoderModel.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    base_encoder.enable_input_require_grads()
    base_encoder.config.use_cache = False 

    train_samples = len(tokenized_dataset["train"])
    total_training_steps = (train_samples // 32) * 3 

    peft_config = AdaLoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, 
        init_r=48,          
        target_r=8,        
        lora_alpha=16,      
        lora_dropout=0.1,   
        target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"], 
        total_step=total_training_steps,
        tinit=int(total_training_steps * 0.1),
        tfinal=int(total_training_steps * 0.2),
        deltaT=10,
        orth_reg_weight=0.1
    )
    
    tunable_encoder = get_peft_model(base_encoder, peft_config)
    for param in tunable_encoder.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    model = ContrastiveModel(encoder=tunable_encoder)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=8,     
        per_device_eval_batch_size=8,      
        gradient_accumulation_steps=4,     
        gradient_checkpointing=True,       
        dataloader_num_workers=4,          
        eval_accumulation_steps=4,         
        optim="adamw_torch_fused",         
        learning_rate=2e-4,                
        num_train_epochs=3,       
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", 
        greater_is_better=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),   
        save_safetensors=False, 
        report_to="none"
    )

    trainer = SupConTrainer(
        model=model, 
        args=training_args, 
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"], 
        data_collator=data_collator,       
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    model.encoder.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()