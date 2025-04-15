import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup
from model import load_model_and_tokenizer
from load_data import get_dataloaders
from tqdm import tqdm
import gc
from torch.optim.lr_scheduler import OneCycleLR


# Configs
EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 5.112181938240301e-07
max_lr = 6.441055391316302e-06
pct_start = 0.10949648796859819
WEIGHT_DECAY = 0.05183162325739168
ADAM_EPS = 2.0561981026966846e-06
GRAD_CLIP = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# State tracking variables
interval_fraction = 0.25
best_model_name = ""

print(f"Using device: {DEVICE}")

def evaluate(model, val_loader):
    model.eval()
    torch.cuda.empty_cache()
    gc.collect()    
    total_loss = 0.0

    with torch.inference_mode():  # More memory-efficient than no_grad
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            # Move inside loop to ensure proper autocast device placement
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # Correct device_type string
            with torch.autocast(device_type='cuda' if DEVICE.type == 'cuda' else 'cpu', 
                              dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                total_loss += outputs.loss.item()

    return total_loss / len(val_loader)


def has_nan_grad(model):
    return any(
        p.grad is not None and not torch.isfinite(p.grad).all()
        for p in model.parameters()
    )

def train():
    # Load model and tokenizer (pad_token automatically handled)
    model, tokenizer = load_model_and_tokenizer("gpt2-medium")
    model.to(DEVICE)
    prev_interval_score = float("inf")

    # Load pre-tokenized dataloaders
    train_loader, val_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        tokenizer_name="gpt2-medium"
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=ADAM_EPS
    )

    # Scheduler
    total_steps = len(train_loader) * EPOCHS
    lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,  # Spend 20% of steps warming up
            anneal_strategy='cos'
        )

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

            try:
                loss = outputs.loss
                if not torch.isfinite(loss):
                    raise ValueError("NaN loss")
            except:
                tqdm.write("NaN or unstable loss — skipping batch")
                continue

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            lr_scheduler.step() 
   
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
            })
            # Intermediate validation (25% of epoch)
            interval_len = int(len(train_loader) * interval_fraction)
            if (step + 1) % interval_len == 0:
                avg_train_loss = epoch_loss / (step + 1)
                val_loss = evaluate(model, val_loader)
                mean_score = (avg_train_loss + val_loss) / 2
                tqdm.write(f"\n25% Interval @ step {step+1} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | Mean: {mean_score:.4f}")

                if mean_score < prev_interval_score:
                    prev_interval_score = mean_score
                    ckpt_folder = f"checkpoints/epoch{epoch+1}_step{step+1}_mean{mean_score:.4f}_run_3"
                    model.save_pretrained(ckpt_folder)
                    tokenizer.save_pretrained(ckpt_folder)
                    tqdm.write(f"Saved improved checkpoint: {ckpt_folder}")
                else:
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        param_group["lr"] = min(param_group["lr"] * 1.15, max_lr)
                    tqdm.write(f"🚀 Boosted LR from {old_lr:.2e} → {param_group['lr']:.2e}")
                    current_lr = optimizer.param_groups[0]['lr']
                    progress_bar.set_postfix(loss=loss.item(), lr=current_lr)


        # Epoch summary
        final_val_loss = evaluate(model, val_loader)
        print(f"\nEpoch {epoch+1} Complete | Final Val Loss: {final_val_loss:.4f}")
        ckpt_folder = f"checkpoints/epoch{epoch+1}_final_run_3"
        model.save_pretrained(ckpt_folder)
        tokenizer.save_pretrained(ckpt_folder)

    print("Training complete!")

if __name__ == "__main__":
    train() 