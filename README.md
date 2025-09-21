# Natural Farming QA baseline Model  🌱

Fine-tune Google's Gemma 3 1B model for answering questions about natural farming practices using Supervised Fine-Tuning (SFT).

## 📋 Overview

This project demonstrates how to fine-tune the `google/gemma-3-1b-it` model on a custom dataset of natural farming questions and answers. The model is trained to provide knowledgeable responses about sustainable and natural farming techniques.

## 🚀 Features

- **Model**: Google Gemma 3 1B Instruction-Tuned
- **Training Method**: Supervised Fine-Tuning (SFT) using TRL library
- **Dataset Format**: JSONL with question-answer pairs
- **Hardware Support**: GPU acceleration with automatic mixed precision
- **Interactive Testing**: Built-in chat interface for model evaluation

## 📋 Requirements

### Hardware
- GPU with at least 8GB VRAM (recommended)
- Google Colab with GPU runtime (T4 or better)

### Software Dependencies
```bash
torch
tensorboard
transformers
datasets
accelerate
evaluate
trl
sentencepiece
huggingface-hub
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/prikshitkverma/Gemma_fine_tuning.git
cd Gemma_fine_tuning
```

2. **Install dependencies**
```bash
pip install -q -U torch tensorboard
pip install -q -U transformers datasets accelerate evaluate trl sentencepiece
```

## 📊 Dataset Format

Your dataset should be in JSONL format with the following structure:

```json
{"question": "What is natural farming?", "answer": "Natural farming is a sustainable agricultural method that..."}
{"question": "How do you prepare compost?", "answer": "To prepare compost, you need organic materials like..."}
```

**Required file**: `nf_dataset_augmented.jsonl`

## 🔧 Configuration

### 1. Hugging Face Authentication

Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens) and replace:

```python
HF_TOKEN = "Enter HF Token Here"  # Replace with your actual token
```

### 2. Model Configuration

```python
base_model = "google/gemma-3-1b-it"
output_dir = "./gemma-natural-farming-qa"
data_file = "/content/nf_dataset_augmented.jsonl"  # Update path as needed
```

### 3. Training Parameters

Key hyperparameters you can adjust:

- `num_train_epochs`: Number of training epochs (default: 3)
- `per_device_train_batch_size`: Batch size per device (default: 2)
- `gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `learning_rate`: Learning rate (default: 2e-5)
- `max_length`: Maximum sequence length (default: 512)

## 🎯 Usage

### Running the Training

1. **Upload your dataset** to the specified path
2. **Update the HF_TOKEN** with your actual token
3. **Run the complete script**:

```python
python train_natural_farming_qa.py
```

### Training Process

The script will:
1. Load and format your dataset
2. Split data into train/test sets (80/20)
3. Load the base Gemma model
4. Configure training parameters
5. Start fine-tuning process
6. Save the trained model
7. Launch interactive testing interface

### Model Testing

After training, the script automatically starts an interactive session:

```
Enter your question (or type 'exit' to quit): What is companion planting?

Generated Answer:
Companion planting is the practice of growing different plants together...
```

## 📁 Project Structure

```
Gemma_fine_tuning/
├── README.md
├── train_natural_farming_qa.py
├── nf_dataset_augmented.jsonl
├── gemma-natural-farming-qa/     # Output directory
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   └── ...
└── runs/                         # TensorBoard logs
```

## 📊 Training Monitoring

Monitor training progress using TensorBoard:

```bash
tensorboard --logdir=runs
```

## 💾 Model Saving and Loading

### Save Model
The model is automatically saved to the `output_dir` after training.

### Load for Inference
```python
from transformers import pipeline

pipe = pipeline("text-generation", 
                model="./gemma-natural-farming-qa", 
                tokenizer=tokenizer)
```

## 🔧 Customization

### Adjusting Training Parameters

For different hardware or dataset sizes:

```python
sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=5,                     # More epochs for larger datasets
    per_device_train_batch_size=1,          # Reduce if GPU memory is limited
    gradient_accumulation_steps=8,          # Increase to maintain effective batch size
    learning_rate=1e-5,                     # Lower LR for more stable training
    max_length=1024,                        # Increase for longer conversations
    # ... other parameters
)
```

### Dataset Modifications

To use different question-answer formats, modify the `format_dataset` function:

```python
def format_dataset(sample):
    return {
        "messages": [
            {"role": "user", "content": sample["your_question_field"]},
            {"role": "assistant", "content": sample["your_answer_field"]}
        ]
    }
```

## 📈 Expected Results

- **Training Time**: ~30-60 minutes on Google Colab T4 GPU
- **Model Size**: ~2GB for the fine-tuned model
- **Performance**: Improved domain-specific responses for natural farming queries

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce `per_device_train_batch_size` to 1
   - Increase `gradient_accumulation_steps` to maintain effective batch size

2. **Dataset Loading Error**
   - Verify JSONL file path and format
   - Check that each line is valid JSON

3. **Authentication Error**
   - Ensure your Hugging Face token has read permissions
   - Verify token is correctly set in the code

### Memory Optimization

For limited GPU memory:
```python
# Enable gradient checkpointing
sft_config.gradient_checkpointing = True

# Use smaller batch size
sft_config.per_device_train_batch_size = 1
sft_config.gradient_accumulation_steps = 8
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **Google** for the Gemma model series
- **Hugging Face** for the transformers and TRL libraries

## 📚 Additional Resources

- [Gemma Model Documentation](https://huggingface.co/google/gemma-3-1b-it)
- [TRL Library Documentation](https://huggingface.co/docs/trl)
- [Hugging Face Fine-tuning Guide](https://huggingface.co/docs/transformers/training)

---

**Note**: This model is for educational and research purposes. Always verify agricultural advice with local experts and extension services.
