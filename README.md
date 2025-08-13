# Text Summarization & Question Answering System

A comprehensive NLP system that combines text summarization using fine-tuned T5 model with question-answering capabilities using DistilBERT.

## 🎯 Project Overview

This project implements an end-to-end text summarization pipeline that:
- Preprocesses and cleans text data
- Fine-tunes a T5-small model for summarization
- Integrates a question-answering system
- Evaluates performance using ROUGE and BERTScore metrics

## 📊 Dataset

The project uses the CNN/DailyMail dataset:

### Dataset Sources:
- **Primary Dataset**: [CNN-DailyMail News Text Summarization](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)


### Dataset Statistics:
- **Training set**: 10,000 samples (sampled from original)
- **Validation set**: 4,000 samples
- **Test set**: 2,000 samples
- **Total Original Size**: 286,817 training pairs, 13,368 validation pairs and 11,487 test pairs
- **Content**: Over 300k unique news articles from CNN and Daily Mail journalists

### Loading the Dataset
```python
from datasets import load_dataset
import pandas as pd

# Option 1: Load from Hugging Face
dataset = load_dataset("ccdv/cnn_dailymail", "3.0.0")

# Option 2: Load from Kaggle (after downloading)
train_data = pd.read_csv("path/to/train.csv")
val_data = pd.read_csv("path/to/validation.csv")  
test_data = pd.read_csv("path/to/test.csv")
```

## 🛠️ Features

### Text Preprocessing
- Lowercase conversion
- HTML tag removal
- URL removal
- Punctuation cleaning
- Stop words filtering (preserving negation words)
- Text normalization

### Model Architecture
- **Base Model**: [T5-small](https://huggingface.co/google-t5/t5-small) - Google's Text-to-Text Transfer Transformer
- **Summarization**: Fine-tuned T5-small model on CNN/DailyMail
- **Question Answering**: [DistilBERT-base-cased-distilled-squad](https://huggingface.co/distilbert-base-cased-distilled-squad)

### Pre-trained Models Available
- **Our Fine-tuned Model**: `t5-Text-Summarizer` (local)
- **Alternative Models**:
  - [ndtran/t5-small_cnn-daily-mail](https://huggingface.co/ndtran/t5-small_cnn-daily-mail)
  - [mrm8488/t5-base-finetuned-summarize-news](https://huggingface.co/mrm8488/t5-base-finetuned-summarize-news)
  - [Falconsai/text_summarization](https://huggingface.co/Falconsai/text_summarization)

### Evaluation Metrics
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
- BERTScore (Precision, Recall, F1)

## 🚀 Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd text-summarization-qa

# Install required packages
pip install -r requirements.txt
```

## 📦 Dependencies

```python
numpy
pandas
seaborn
nltk
transformers
torch
datasets
rouge-score
bert-score
tqdm
matplotlib
```


## 📈 Model Performance

### Training Configuration
- **Learning Rate**: 2e-5
- **Batch Size**: 2 (train/eval)
- **Epochs**: 5
- **Max Input Length**: 95th percentile of article lengths
- **Max Target Length**: 95th percentile of summary lengths

### Evaluation Results
The model is evaluated using:
- **ROUGE Scores**: Measures overlap between generated and reference summaries
- **BERTScore**: Semantic similarity using BERT embeddings

## 📊 Visualizations

The project includes:
- Token length distribution plots
- Training and evaluation loss curves
- Performance metrics visualization

## 🏗️ Project Structure

```
text-summarization-qa/
├── README.md
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── models/
│   └── t5-Text-Summarizer/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


