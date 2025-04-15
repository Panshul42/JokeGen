"""
Model training with bayesian hyperparameter optimization
"""
import optuna
import torch
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_cosine_schedule_with_warmup
from model import load_model_and_tokenizer
from load_data import get_dataloaders
from tqdm import tqdm
import gc
from torch.optim.lr_scheduler import OneCycleLR


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

# State tracking variables
interval_fraction = 0.75
best_model_name = ""

# Data loaders

train_loader_cache, val_loader_cache = get_dataloaders(
        batch_size=BATCH_SIZE,
        tokenizer_name="gpt2-medium"
    )


print(f"Using device: {DEVICE}")

def objective(trial):
    return train_with_params(
        trial=trial,
        learning_rate=trial.suggest_float("learning_rate", 2e-7, 8e-7, log=True),
        max_lr=trial.suggest_float("max_lr", 5e-6, 2e-5, log=True),
        weight_decay=trial.suggest_float("weight_decay", 0.008, 0.05, log=True),
        grad_clip=trial.suggest_categorical("grad_clip", [0.4, 0.45, 0.475, 0.5, 0.535, 0.55]),
        pct_start=trial.suggest_float("pct_start", 0.1, 0.18),
        adam_eps=trial.suggest_float("adam_eps", 2e-6, 8e-6, log=True),
        fast_mode=True
    )

def log_trial_result(trial_number, mean_score, val_loss, train_loss, params, filepath="trial_logs.txt"):
    with open(filepath, "a") as f:
        f.write(f"[Trial {trial_number}] | Mean score: {mean_score:.4f} | Val loss: {val_loss:.4f} | Train loss: {train_loss:.4f} | Params: {params}\n")

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

def train_with_params(trial=None, 
                      learning_rate=2e-7, 
                      max_lr=2e-6,
                      weight_decay=0.01,
                      grad_clip=0.4,
                      pct_start=0.16,
                      adam_eps=1e-6,
                      fast_mode=True,
                      train_loader = train_loader_cache,
                      val_loader = val_loader_cache):

    EPOCHS = 2 if not fast_mode else 1
    BATCH_SIZE = 128

    model, tokenizer = load_model_and_tokenizer("gpt2-medium")
    model.to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=adam_eps
    )

    total_steps = len(train_loader) * EPOCHS
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy='cos'
    )

    interval_fraction = 0.75

    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        if not torch.isfinite(loss):
            tqdm.write("Exploding loss detected — terminating trial.")
            if trial is not None:
                raise optuna.exceptions.TrialPruned()
            else:
                return float("inf")  # Safe fallback if not using Optuna

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        lr_scheduler.step()

        epoch_loss += loss.item()

        progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
        })

        interval_len = int(len(train_loader) * interval_fraction)
        if (step + 1) % interval_len == 0:
            avg_train_loss = epoch_loss / (step + 1)
            val_loss = evaluate(model, val_loader)
            mean_score = ((avg_train_loss + val_loss) / 2) + abs(avg_train_loss - val_loss)/2  # we penalize for large differences

            # Report intermediate results to Optuna
            if trial is not None:
                trial.report(mean_score, step)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if fast_mode:
                print(f"[Trial {trial.number}] Train loss: {avg_train_loss:.4f} | Val loss: {val_loss:.4f} | Mean score: {mean_score:.4f} | Params: {trial.params}")
                log_trial_result(trial.number, mean_score, val_loss, avg_train_loss, trial.params)
                return mean_score

    # Final evaluation
    final_val_loss = evaluate(model, val_loader)
    return final_val_loss

study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
)
study.optimize(objective, n_trials=20)

print("Best Trial:")
print(study.best_trial.params)
print(study.best_trial.value)