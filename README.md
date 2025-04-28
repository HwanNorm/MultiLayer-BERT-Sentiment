# MultiLayer-BERT-Sentiment
# Multi-Layer BERT for Sentiment Analysis

![Model Performance](https://raw.githubusercontent.com/yourusername/repo-name/main/model_performance.png)

## Project Overview
This project implements an enhanced sentiment analysis model by leveraging multi-layer features from BERT transformer blocks. Instead of using only the final layer's representation, this approach extracts and combines features from all encoder layers to create a more robust text representation for sentiment classification.

## Key Features
- **Multi-layer Feature Extraction**: Extracts embeddings from all four BERT encoder layers
- **Feature Fusion**: Combines layer outputs using global average pooling and concatenation
- **Deep Classifier Network**: Implements a multi-layer neural network with batch normalization and dropout
- **IMDB Dataset**: Trained and evaluated on the widely-used IMDB movie reviews dataset

## Model Architecture

### BERT Base
- Using `small_bert/bert_en_uncased_L-4_H-128_A-2`
- 4 Transformer encoder blocks
- 128 hidden dimensions
- 2 attention heads

### Feature Extraction Strategy
Each layer in BERT captures different semantic information:
- Lower layers: Syntactic and local contextual information
- Higher layers: More abstract, task-relevant features

The model performs global average pooling on each layer's output and concatenates them to form a comprehensive feature vector.

### Classifier Architecture
```
Sequential(
  InputLayer(input_shape=(512,))  # 4 layers × 128 hidden size
  
  Dense(256, activation=None)
  BatchNormalization()
  Activation('relu')
  Dropout(0.4)
  
  Dense(128, activation=None)
  BatchNormalization()
  Activation('relu')
  Dropout(0.3)
  
  Dense(64, activation=None)
  BatchNormalization()
  Activation('relu')
  Dropout(0.2)
  
  Dense(1, activation='sigmoid')
)
```

## Performance Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 72.62% |
| Test Loss | 0.5332 |
| Training Time | ~150s/epoch |

### Learning Curves
The model shows steady improvement in accuracy over training epochs, with validation accuracy consistently above training accuracy, indicating good generalization without overfitting.

## Sample Predictions

| Review | Prediction | Confidence |
|--------|------------|------------|
| "This movie was fantastic! Great acting and storyline." | Positive | 91.15% |
| "The film was excellent, with amazing performances by the entire cast." | Positive | 99.54% |
| "Terrible film. Complete waste of time and money." | Negative | 69.18% |
| "I hated this movie. The plot made no sense and the acting was terrible." | Negative | 96.12% |

## Implementation Details

### Data Preprocessing
```python
def map_function(text, label):
    # Preprocess text using BERT preprocessing
    preprocessed = preprocessor(text)

    # Extract embeddings from each encoder block
    encoder_outputs = bert_model(preprocessed)['encoder_outputs']

    # Apply pooling to each layer's output
    pooled_outputs = []
    for layer_output in encoder_outputs:
        # Apply global average pooling
        pooled = tf.reduce_mean(layer_output, axis=1)
        pooled_outputs.append(pooled)

    # Combine the embedding feature vectors from all encoder blocks
    combined_output = tf.concat(pooled_outputs, axis=-1)

    return combined_output, label
```

### Training Configuration
- Optimizer: Adam with learning rate 5e-4
- Loss: Binary Cross Entropy
- Batch Size: 64
- Early Stopping: 2 epochs patience monitoring validation accuracy

## How to Use

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/bert-multilayer-sentiment.git
cd bert-multilayer-sentiment

# Install dependencies
pip install tensorflow tensorflow_datasets tensorflow_hub tensorflow_text matplotlib
```

### Training
```bash
python train.py
```

### Inference
```python
# Sample code for making predictions
def predict_sentiment(texts):
    # Make sure texts is a list
    if isinstance(texts, str):
        texts = [texts]

    # Preprocess texts
    preprocessed = preprocessor(texts)

    # Get BERT outputs
    bert_output = bert_model(preprocessed)
    encoder_outputs = bert_output['encoder_outputs']

    # Pool and combine outputs from all layers
    pooled_outputs = []
    for layer_output in encoder_outputs:
        pooled = tf.reduce_mean(layer_output, axis=1)
        pooled_outputs.append(pooled)

    combined = tf.concat(pooled_outputs, axis=-1)

    # Make prediction
    return classifier.predict(combined)

# Usage example
sample_review = "This movie was fantastic! Great acting and storyline."
prediction = predict_sentiment(sample_review)
sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
confidence = prediction[0][0] if sentiment == "Positive" else 1 - prediction[0][0]
print(f"Prediction: {sentiment} (confidence: {confidence:.4f})")
```

## Technical Challenges and Solutions

### Challenges
1. **Memory Management**: Processing full BERT outputs for larger datasets
2. **Training Time**: Fine-tuning transformer models is computationally expensive
3. **Layer Integration**: Effectively combining information from different layers
4. **Overfitting**: Preventing overfitting with the large number of parameters

### Solutions
1. Implemented batch processing and prefetching for efficient data handling
2. Used a smaller BERT variant (L-4, H-128, A-2) and applied early stopping
3. Applied global average pooling to reduce dimensionality before concatenation
4. Incorporated dropout and batch normalization in the classifier architecture
