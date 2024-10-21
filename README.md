# Handwriting Recognition Using KNN and Neural Networks with PCA

## Overview
This project focuses on predicting handwritten digits, with a goal to assist schools in identifying students who may need help with their motor skills. The approach involves building and comparing two machine learning models: a K-Nearest Neighbors (KNN) classifier and a Neural Network (NN). Dimensionality reduction is applied using Principal Component Analysis (PCA) to optimize model performance and reduce the feature space. The project evaluates both models based on their accuracy, precision, recall, and F1-score, providing insight into which model is most suitable for the task.

## Models Used
- **K-Nearest Neighbors (KNN)**: A simple and intuitive classification algorithm that classifies data points based on the 'k' nearest neighbors.
- **Neural Network (NN)**: A more advanced model, capable of capturing complex non-linear relationships in the data through multiple layers and neurons.

## Dataset
The dataset consists of pixel values representing handwritten digits, labeled from 0 to 9. Each row in the dataset represents a single handwritten sample, with the columns containing pixel values. Principal Component Analysis (PCA) is applied to reduce the number of features (dimensions) while retaining 95% of the dataset’s variance.

- **Number of samples**: 42,000 handwritten digit samples.
- **Number of features**: Initially 45 pixel features, reduced to 26 using PCA.

## Problem Statement
The objective is to build a model that can accurately predict which digit is represented by a given handwriting sample. By automating this process, the school can use this technology to identify students who may need additional support with their motor skills.

## Methodology
1. **Data Preprocessing**: 
   - The pixel values are normalized to ensure they are on the same scale.
   - PCA is applied to reduce the dimensionality of the data, speeding up the computation and reducing overfitting.
   
2. **Model Training**:
   - **KNN**: A KNN model is trained on the PCA-transformed data. The `k=5` neighbors were used for this task.
   - **Neural Network**: A feedforward neural network with two hidden layers (128 and 64 neurons) was trained on the same PCA-transformed data.
   
3. **Model Evaluation**:
   - Both models are evaluated on the test set using accuracy, precision, recall, and F1-score.

### Results
- **KNN Accuracy**: 64%
- **Neural Network Accuracy**: 69%

While the Neural Network outperforms KNN, both models provide valuable insights into recognizing patterns in handwritten digits.

### Key Performance Metrics:
- **KNN Model**:
  - Precision: 0.63 (macro average)
  - Recall: 0.63 (macro average)
  - F1-Score: 0.63 (macro average)
  
- **Neural Network**:
  - Precision: 0.70 (macro average)
  - Recall: 0.69 (macro average)
  - F1-Score: 0.69 (macro average)
  
### Challenges
- **KNN**:
  - High-dimensional data can be computationally expensive for KNN due to the distance calculations.
  - Reducing dimensionality using PCA helped improve speed but the model still struggled to correctly classify more complex digits, like 7 and 9.
  
- **Neural Network**:
  - Neural Networks require significant computational resources, and hyperparameter tuning is crucial for optimizing performance.
  - The model showed better performance but took longer to train due to its complexity.

## Installation
To run this project locally, you will need to clone the repository and install the necessary dependencies.

```bash
https://github.com/dendarko/handwriting-recognition-knn-neuralnet.git
cd handwriting-recognition
pip install -r requirements.txt
