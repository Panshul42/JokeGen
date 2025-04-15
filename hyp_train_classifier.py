import optuna
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler
from transformers import get_scheduler
from load_data_classif import get_classification_dataloaders, load_model_and_tokenizer
from tqdm import tqdm

LOG_FILE = "classif_trial_logs.txt"

def log_trial_result(trial_number, val_loss, val_acc, params, filepath=LOG_FILE):
    with open(filepath, "a") as f:
        f.write(f"[Trial {trial_number}] "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Params: {params}\n")

def objective(trial):
    config = {
        "epochs": 2,
        "lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
        "clip_grad_norm": trial.suggest_float("clip_grad_norm", 0.1, 2.0),
        "log_every": 100,
        "batch_size": 64,
    }

    train_loader, val_loader = get_classification_dataloaders(batch_size=config["batch_size"])
    model, tokenizer = load_model_and_tokenizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scaler = GradScaler(device_type='cuda')

    optimizer = AdamW(model.parameters(), 
                      lr=config["lr"],
                      weight_decay=config["weight_decay"],
                      betas=(0.9, 0.999),
                      fused=True)

    num_training_steps = config["epochs"] * len(train_loader)
    scheduler = get_scheduler("linear", optimizer, 
                              num_warmup_steps=config["warmup_steps"], 
                              num_training_steps=num_training_steps)

    loss_fn = CrossEntropyLoss()
    model.train()
    for epoch in range(config["epochs"]):
        for batch in tqdm(train_loader, desc=f"Trial Epoch {epoch+1}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).long().view(-1)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)

            if not torch.isfinite(loss):
                raise optuna.exceptions.TrialPruned()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).long().view(-1)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)

            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total

    log_trial_result(trial.number, avg_val_loss, accuracy, trial.params)
    trial.report(avg_val_loss, epoch)
    return avg_val_loss

def main():
    # clear log file before starting
    open(LOG_FILE, "w").close()
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("\nBest Trial:")
    print(study.best_trial.params)
    print(f"Best Validation Loss: {study.best_trial.value:.4f}")

if __name__ == "__main__":
    main()