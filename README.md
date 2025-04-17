# JokeGen: Computational Humor Generation and Classification

## Project Overview
JokeGen is a computational humor project that leverages transformer architectures for both joke generation and humor classification. The project consists of two main components:

1. **Joke Generator**: A fine-tuned GPT-2-Medium model that completes or writes jokes.
2. **Humor Classifier**: A binary classification model that determines whether a given text is funny or not

## Data Sources
- **Joke Generation**: Combined dataset of ~400,000 jokes from Reddit and other sources
- **Humor Classification**: The ColBERT humor detection dataset with 100,000 funny and 100,000 non-funny text samples

## Architecture

### Joke Generator
- Pre-trained GPT-2-Medium model (355M parameters)
- Fine-tuned with autoregressive language modeling objective
- Optimized with OneCycleLR learning rate scheduling
- Trained using mixed precision (FP16) for efficiency

### Humor Classifier
- Custom transformer-based binary classification architecture
- Designed for effective humor pattern recognition
- Optimized with AdamW and linear warmup scheduler
- Trained using gradient scaling and mixed precision

## Technical Optimizations

### Memory Efficiency
- **4-bit Quantization**: Implemented BitsAndBytes 4-bit quantization to reduce memory footprint
- **Gradient Checkpointing**: Enabled for lower memory usage during training
- **Smart Data Collation**: Custom collator for handling variable-length sequences efficiently

### Training Performance
- **Mixed Precision Training**: Utilized torch.amp for faster computation with minimal accuracy loss
- **Fused Optimizer**: Implemented fused AdamW for improved training throughput
- **Gradient Clipping**: Applied norm-based gradient clipping to stabilize training
- **Dynamic Batch Sorting**: Sorted sequences by length to minimize padding waste

### Hyperparameter Optimization
- **Bayesian Optimization**: Used Optuna for principled hyperparameter search
- **Early Pruning**: Implemented MedianPruner to terminate underperforming trials
- **Cosine Annealing**: Applied cosine annealing with warmup for learning rate scheduling
- **Intermediate Evaluation**: Conducted periodic validation during training for early feedback

## Results

All training was performed on an Nvidia RTX 4090. Average training time was under five minutes.

### Joke Generator
- Best validation loss achieved was 2.27.
- Capable of generating contextually coherent joke completions from prompts
- Sample outputs:

```
📝 Sample: Why did the chicken cross the road? Because he was really drunk.

📝 Sample: I went to the doctor today and guess what he told me? I'd see him next day.

📝 Sample: Why did the robot go to therapy? Because he was busy trying to raise his brain to have a life.
```

### Humor Classifier
- **Validation Accuracy**: 97.46% after five epochs
- **High Precision**: Effectively distinguishes between humorous and non-humorous content
- **Fast Inference**: Optimized for quick classification of short text snippets

## Technical Requirements
See `requirements.txt` for full dependencies. Core requirements include:
- PyTorch
- Transformers
- Datasets
- Optuna
- BitsAndBytes
- Accelerate

## Future Work
- Explore larger model architectures for improved generation quality
- Implement few-shot learning techniques for more versatile humor recognition
- Develop an ensemble approach combining multiple humor classification models
- Create an interactive demo application for real-time joke generation

## Conclusion
This project demonstrates the successful application of state-of-the-art NLP techniques to the challenging domain of computational humor. Through careful optimization and hyperparameter tuning, we've created models that can both generate humorous content and accurately classify text by its humor potential.