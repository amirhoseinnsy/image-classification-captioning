# 📸📝 Image Captioning with Encoder–Decoder Networks

This project implements an image captioning system that generates natural language descriptions for images. The model is built using a CNN-based encoder to understand image features and an RNN-based decoder to generate textual captions.

## Table of Contents
- [Task Overview](#-task-overview)
- [Data Preparation](#️-data-preparation)
- [Model Architecture](#️-model-architecture)
  - [Encoder](#encoder)
  - [Decoder](#decoder)
- [Training Setup](#training-setup)
- [Improvements & Challenges](#-improvements--challenges)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Decoding Strategies](#decoding-strategies)
  - [Teacher Forcing](#teacher-forcing)
  - [Attention Mechanism](#attention-mechanism)
- [Experiments](#-experiments)
- [Key Learnings](#-key-learnings)
- [Installation](#installation)
- [Usage](#usage)

---

## 📌 Task Overview

The primary goal is to build and train an encoder–decoder model capable of generating descriptive captions for a given image.

- **Dataset**: The model is trained on standard image captioning datasets such as **Flickr8k**, **MS COCO**, or **Flickr30k (Karpathy splits)**.
- **Ground Truth**: Each image in the dataset is associated with multiple human-written reference captions, which are used for training and evaluation.

---

## ⚙️ Data Preparation

Proper data handling is critical for training the model effectively.

1.  **Dataset Splitting**: The dataset is downloaded and split into training, validation, and test sets.
2.  **Caption Tokenization**:
    - Special `<start>` and `<end>` tokens are added to each caption to signify the beginning and end of a sentence.
    - A vocabulary is constructed from all unique words in the training captions.
    - Captions are padded or truncated to ensure a consistent length for batch processing.

---

## 🏗️ Model Architecture

The model follows a standard encoder-decoder architecture.

### Encoder

The encoder is responsible for extracting high-level visual features from the input image.
- **Model**: A pretrained **ResNet-101** CNN is used.
- **Feature Extraction**: The output from the last convolutional layer (a `7×7×2048` feature map) is used as the image representation. These feature vectors are then passed to the decoder.

### Decoder

The decoder is an RNN-based sequence generator that produces the caption word by word.
- **Model**: An **LSTM** (or optionally a **GRU**) network.
- **Word Embeddings**: An embedding layer converts word tokens into dense vectors. This layer can be initialized with pretrained embeddings like **word2vec** or **GloVe** for improved performance.
- **Generation Process**: The decoder takes the image features as its initial state and generates the caption one word at a time until an `<end>` token is produced.

---

## Training Setup

The model is trained end-to-end with the following configuration:
- **Loss Function**: Cross-Entropy Loss.
- **Optimizer**: Adam.
- **Hyperparameters**:
    - Embedding Dimension: `512`
    - Batch Size: `60`
    - Learning Rate: `1e-5`
- **Regularization**: Early stopping is used to prevent overfitting by monitoring validation loss.

---

## 🔍 Improvements & Challenges

Several techniques were implemented to enhance the model's performance and address common challenges in sequence generation.

### Evaluation Metrics
- The **BLEU (Bilingual Evaluation Understudy)** score is used to quantitatively measure the quality of the generated captions by comparing them against the reference captions.

### Decoding Strategies
- **Greedy Decoding**: A naive approach where the most probable word is chosen at each timestep.
- **Beam Search**: A more advanced strategy that explores multiple probable sequences of words at each step. This leads to higher-quality captions and helps avoid repetitive phrases.

### Teacher Forcing
- To stabilize the training process, **teacher forcing** is used. During training, the decoder is fed the ground-truth tokens from the previous timestep instead of its own predictions, which helps it learn the sequence structure more effectively.

### Attention Mechanism
- A visual attention mechanism was added to the decoder. This allows the model to focus on the most relevant regions of the image when generating each word.
- **Benefit**: Attention not only improves caption accuracy but also makes the model more interpretable, as the focus regions can be visualized as heatmaps.

---

## 📊 Experiments

A series of experiments were conducted to evaluate the impact of different components:
1.  **Baseline Model**: A simple ResNet-101 encoder + LSTM decoder.
2.  **Pretrained Embeddings**: The baseline was improved by initializing the word embedding layer with pretrained word2vec and GloVe vectors.
3.  **Beam Search Decoding**: The generation quality was enhanced by replacing greedy decoding with beam search during inference.
4.  **Attention Mechanism**: The final model incorporated the attention mechanism.

The performance of each model variant was compared using the BLEU score.

---

## 🔬 Key Learnings

- **Tokenization**: The use of `<start>` and `<end>` markers is crucial for signaling the boundaries of a caption, enabling the model to learn when to start and stop generating words.
- **Decoding Strategy**: Beam Search significantly improves the fluency and coherence of generated captions compared to greedy decoding.
- **Attention**: The attention mechanism is a powerful addition that allows for more accurate and context-aware captions while also providing valuable insights into the model's decision-making process.
- **Evaluation**: The BLEU score provides a standardized and quantitative metric for comparing the performance of different image captioning models.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/image-captioning.git
    cd image-captioning
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```


## Usage

run different functions of main.py
