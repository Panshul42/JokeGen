import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel
from load_data import get_dataloaders
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def generate_text(
    prompt,
    model,
    tokenizer,
    max_new_tokens=64,
    temperature=0.85,
    top_k=50,
    top_p=0.95
):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    generated = input_ids

    for _ in range(max_new_tokens):
        if generated.size(1) >= model.config.n_positions:
            break

        with torch.amp.autocast(device_type=DEVICE.type):
            outputs = model(generated)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            next_token_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_k_vals = torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[next_token_logits < top_k_vals] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                for b in range(next_token_logits.size(0)):
                    next_token_logits[b, sorted_indices[b][sorted_indices_to_remove[b]]] = float('-inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id and generated.shape[1] >= 50:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

@torch.no_grad()
def run_complete_validation(model, val_loader):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(val_loader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with torch.amp.autocast(device_type=DEVICE.type):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs.loss.item()
    return total_loss / len(val_loader)

if __name__ == "__main__":
    import os
    import datetime
    from datasets import load_dataset

    base_dir = "./checkpoints"
    os.makedirs("results", exist_ok=True)

    from load_data import get_dataloaders
    _, val_loader = get_dataloaders(batch_size=8, tokenizer_name="gpt2", val_ratio=0.1)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"results/eval_log_{timestamp}.txt"

    prompts = [
        "Why",
        "Did",
        "I",
        "That big",
        "Knock knock...",
        "Your momma so...",
        "Why did the chicken cross the road?",
        "The bartender looked at the priest and said...",
        "Here's the real reason why aliens haven't contacted us yet...",
        "If I had a superpower, it would be...",
        "Back in my day, we didn't have phones. We had...",
        "Why did your mom visit the therapist?",
        "What is the meaning of life?",
        "I went to the doctor today and guess what he told me?"
    ]

    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write("Checkpoint Evaluation Log\n")
        f_out.write("="*60 + "\n\n")

        for folder in sorted(os.listdir(base_dir)):
            ckpt_path = os.path.join(base_dir, folder)
            if not os.path.isdir(ckpt_path):
                continue

            try:
                tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
                model = GPT2LMHeadModel.from_pretrained(
                    ckpt_path,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                model.eval()

                val_loss = run_complete_validation(model, val_loader)
                samples = [generate_text(prompt, model=model, tokenizer=tokenizer) for prompt in prompts]

                log_block = f"{folder}\n Val Loss: {val_loss:.4f}\n"
                for idx, sample in enumerate(samples, 1):
                    log_block += f"Sample {idx} Output:\n{sample}\n"
                log_block += "-"*60 + "\n"

                print(log_block)
                f_out.write(log_block)

            except Exception as e:
                err_block = f"Failed on {folder}: {e}\n" + "-"*60 + "\n"
                print(err_block)
                f_out.write(err_block)

    print(f"\nAll results written to: {output_file}")
