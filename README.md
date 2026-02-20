# Project | Deep Learning: Image Classification using CNN and Transfer Learning

## Table of Content
- [Project Overview](#Project-Overview)
- [Key Features](#Key-Features)
- [Main Insights](#Main-Insights)
- [Tools & Techniques](#Tools-&-Techniques)
- [Project Structure](#Project-Structure)

## Project Overview
In this project, we will first build a **Convolutional Neural Network (CNN)** model from scratch to classify images from a given dataset into predefined categories. Then, we will implement a **transfer learning approach** using a pre-trained model. Finally, students will **compare the performance** of the custom CNN and the transfer learning model based on evaluation metrics and analysis.
The dataset we used is the public dataset CIFAR-10, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. You can download the dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Key Features
- **Data Preprocessing**
    - Data loading and preprocessing (if necessary, e.g., normalization, resizing, augmentation).
    - Create visualizations of some images and labels to see the queality of the dataset.
- **Model Architecture**
    - Designed two CNN architectures for image classification of the CIFAR-10 dataset. A simple architecture with only a few layers and a deeper architecture with more layers
    - Including convolutional layers, pooling layers, and fully connected layers.
    - Comparing both architectures.
- **Model Training**
    - Trained the CNN model using appropriate optimization techniques (e.g., Adam).
    - Utilized techniques such as early stopping and reduce learning rate on plateau to prevent overfitting.
- **Model Evaluation**  
    - Evaluated the trained model with the separated validation set.
    - Plotted all metrics such as accuracy, precision, recall, and F1-score.
    - Visualized the confusion matrix to understand model performance across different classes.
- **Transfer Learning**  
    - Performed transfer learning with the pre-trained model [EfficientNetB0](https://keras.io/api/applications/efficientnet/efficientnet_models/#efficientnetb0-function)
    - Trained and evaluated two difference transfer learning models.
        - retrain none of the pre-trained layers
        - retrain the last 20 % of the pre-trained layers
    - Compared the performance between these two and our custom CNN models.
        
## Main Insights
- **base CNN architecture**
    - Doesn't take much time, but the result isn't good.  
        -  ~ 69% test accuracy
- **deeper CNN architecture**
    - Takes a lot of time to get a better result.  
        -  ~ 80% test accuracy
- **pre-trained model EfficientNetB0 without retraining any layers**
    - Very fast in building, good result.  
        - ~ 92% test accuracy
        - ~ 8 minutes training time
- **pre-trained model EfficientNetB0 with retraining the last 20% of layers**
    - Very fast in building, best result.  
        - ~ 95% test accuracy
        - ~ 8 minutes training time
**Using a pretrained model will save time and it will bring you a good result. Retraining some layers will improve your model, but be aware of catastrophical forgetting!**

## Tools & Techniques
- **Programming Languages**: Python (Pandas, NumPy, Matplotlib, Seaborn, tensorflow, sklearn).
- **Programming Environment**: Google Colab Pro
- **Data Sources**: CIFAR-10 dataset from the library tensorflow.keras.

## Project Structure
- `CNN_model.ipynb`: Jupyter Notebook with the data preprocessing and both CNN models (base CNN and deeper CNN) we created from scratch.
- `EfficientNetB0_model.ipynb`: Jupyter Notebook with the data preprocessing and both transfer learning models (with 20% retraining and without retraining).
- `presentation-cnn-and-transfer-learning.pdf`: PDF presentation summarizing the conclusions.