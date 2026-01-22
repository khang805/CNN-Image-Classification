# CIFAR-10 Image Classification with Convolutional Neural Networks (CNN) :

This project involves the design, implementation, and evaluation of a deep Convolutional Neural Network (CNN) for the CIFAR-10 dataset. The system explores advanced regularization techniques like Batch Normalization and
Dropout, followed by an extensive ablation study and feature map visualization to interpret the model's internal learning process.

# 🚀 Key Features
## Hugging Face Integration: 
Seamless data loading using the datasets library for the CIFAR-10 collection.

## Deep Architecture:
A 7-layer CNN designed with optimized filters to extract features from 32x32 RGB images.

## Regularization: 
Implementation of Dropout and Batch Normalization to ensure robust generalization and prevent overfitting.

## Interpretability:
Visualization tools to extract and plot feature maps, showing how the network "sees" edges, textures, and class-specific shapes.

## Comprehensive Ablation Study:
Systematic testing of Learning Rate, Batch Size, Filter Count, and Network Depth to identify the optimal configuration.

# 📂 Repository Structure

```
├── CNN.py                 # Core implementation (Preprocessing, Training, and Evaluation)
├── cnn_training.png       # Visualization of training/validation loss and accuracy
├── feature_maps.png       # Visual representation of learned convolutional filters
├── ablation_results.csv   # Log of hyperparameter tuning experiments
└── README.md              # Project documentation
```

# ⚙️ Installation & Setup

## 1. Prerequisites
Environment: Python 3.8+ or Google Colab (T4 GPU recommended).

Dataset: CIFAR-10 (60,000 images, 10 classes).

## 2. Install Dependencies

pip install -q \
  tensorflow \
  datasets \
  matplotlib \
  seaborn \
  scikit-learn \
  numpy


#  🛠️ Technical Workflow
  
## 1. Dataset Preparation:
Images are loaded from Hugging Face, normalized to a $[0, 1]$ range, and labels are transformed using One-Hot Encoding.

## 2. Model Architecture
The model utilizes a stack of convolutional layers with increasing filter sizes (e.g., 32, 64, 128). Max-pooling layers reduce spatial dimensions, while Dense layers with Softmax activation perform the final
classification across the 10 categories.

## 3. Feature Map Visualization
The system includes a diagnostic pipeline to pass a single image through the trained network and capture the activations of each convolutional layer.

Early Layers: Capture high-frequency details like edges and color gradients.

Deep Layers: Capture complex, abstract shapes relevant to specific classes (e.g., wheels of a car, wings of a bird).

## 📊 Performance & Evaluation

### Results Table
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline CNN** | 0.72 | 0.71 | 0.70 | 0.70 |
| **Optimized CNN** | **0.81** | **0.80** | **0.79** | **0.80** |


## Ablation Study Findings
### Learning Rate: 
0.001 provided the most stable convergence.

### Batch Size:
32 was identified as the optimal balance between speed and accuracy.

### Filters: 
Increasing to 64 filters in the primary layers significantly improved feature extraction.

### Depth:
A 5-layer depth provided the best trade-off between performance and computational cost.


## 🎯 Conclusion
This implementation demonstrates the effectiveness of CNNs in visual recognition tasks. Through rigorous hyperparameter tuning and regularization, the model achieves high accuracy on the CIFAR-10 test set. The project
highlights that while deeper networks offer more representational power, careful management of the learning rate and dropout is essential for training stability.


## 🎓 Author
M Abdurrahman Khan National University of Computer and Emerging Sciences (FAST), Pakistan Contact: {i221148}@nu.edu.pk
