import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import get_scheduler
from torch.cuda.amp import autocast, GradScaler
from load_data_classif import get_classification_dataloaders
from transformer_classifier import load_model_and_tokenizer
import os

def train_classifier(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scaler = GradScaler()

    optimizer = AdamW(model.parameters(), 
                      lr=config["lr"],
                      weight_decay=config["weight_decay"],
                      betas=(0.9, 0.999),
                      fused=True)

    num_training_steps = config["epochs"] * len(train_loader)
    scheduler = get_scheduler("linear", optimizer, 
                              num_warmup_steps=config["warmup_steps"], 
                              num_training_steps=num_training_steps)
    global_step = 0

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        interval_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)
        loss_fn = CrossEntropyLoss(label_smoothing=0.05)

        for batch_idx, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True).long().view(-1)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            interval_loss += current_loss

            preds = torch.argmax(logits, dim=1)
            batch_correct = (preds == labels).sum().item()
            correct += batch_correct
            total += labels.size(0)

            if global_step % config["log_every"] == 0 and global_step > 0:
                avg_loss = interval_loss / config["log_every"]
                accuracy = 100 * batch_correct / labels.size(0)
                val_loss, val_acc = evaluate_classifier(model, val_loader, device)
                
                print(f"\nStep {global_step}:")
                print(f"  Train Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                interval_loss = 0.0

            loop.set_postfix(loss=current_loss, acc=100 * batch_correct / labels.size(0))
            global_step += 1

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Avg Train Loss: {avg_epoch_loss:.4f}")
        print(f"  Train Accuracy: {epoch_acc:.2f}%")     

        val_loss, val_acc = evaluate_classifier(model, val_loader, device)
        print(f"  Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")


def evaluate_classifier(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    loss_fn = CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


# ===============================
# Example `main()` to run training
# ===============================
def main():
    config = {
        "epochs": 5,
        "lr": 0.000084085119953765064,
        "weight_decay": 0.06511038522163902,
        "warmup_steps": 261,
        "clip_grad_norm": 1.8000000000000003,
        "log_every": 1000,
        "batch_size": 128,
    }

    train_loader, val_loader = get_classification_dataloaders(batch_size=config["batch_size"])
    model, tokenizer = load_model_and_tokenizer()


    print("🚀 Starting training...")
    train_classifier(model, train_loader, val_loader, config)

    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "classifier_model_adv_e15_new.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"💾 Model saved to {checkpoint_path}")

if __name__ == "__main__":
    main()