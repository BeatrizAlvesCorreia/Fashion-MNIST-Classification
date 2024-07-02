# Fashion-MNIST-Classification
This project involves building and evaluating machine learning models to classify images from the Fashion MNIST dataset. The dataset includes 70,000 grayscale images of 10 different types of clothing items, divided into training and test sets.
Created by: Beatriz Correia Paulino, Luís Pereira, João Fragoso

## Introduction
In this project, we develop and compare the performance of three types of neural network models:
- A multiclass classification model: Classify images into one of the 10 specific fashion categories.
- A binary classification model: Classify images into two broad categories: "Clothing" and "Footwear and Bags."
- Binarization of the outputs of the multiclass model: Use the outputs of the multiclass model to classify images into the binary categories.
The goal is to assess the effectiveness of each approach in accurately classifying fashion items and to understand the challenges associated with each.

This challenge is separated into 2 scripts:
- MultiClass_Model.ipynb
- Binary_Model.ipynb

Both parts of the challenge focus on different aspects of image classification using the Fashion MNIST dataset. Part 1 deals with multi-class classification, while Part 2 addresses binary classification. Both parts include data preprocessing, model creation, training, and evaluation using TensorFlow and Keras, with visualization of data and results.

## Challenges
- Multiclass Classification: Some fashion items are visually similar (e.g., T-shirt/top and Shirt), making it difficult for the model to distinguish between them.
- Class Imbalance: The distribution of classes might not be uniform, which can affect the model's performance.
- Generalization: Ensuring the model performs well on unseen data (test set) is crucial.

## Data 
This project utilizes the Fashion MNIST dataset, a popular benchmark dataset for machine learning and computer vision tasks. Fashion MNIST is a dataset of 70,000 grayscale images, each measuring 28x28 pixels, categorized into 10 different classes representing various types of clothing items. Specifically, it includes 60,000 training images and 10,000 testing images, ensuring ample data for both training and evaluating models. Each image corresponds to one of the following classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot. This dataset serves as a direct drop-in replacement for the original MNIST dataset of handwritten digits, providing a more challenging and diverse set of data points while maintaining the same image size and structure. The Fashion MNIST dataset is widely used for testing new algorithms and techniques in image classification, making it an excellent choice for this project.

## Models
We created a Sequential model with TensorFlow’s Keras API with the following layers:
- Flatten Layer: Converts the 28x28 pixel input images into a 1D array of 784 elements.
- Dense Layers: Two hidden layers with 32 neurons each and ReLU activation functions.
- Output Layer: A dense layer with a number of neurons equal to the number of classes (N_CLASSES) and a softmax/sigmoid activation function for multiclass and binary approach respectively.

The model is compiled the following way:
- Optimizer: Adam optimizer with a learning rate of 0.001, which is efficient for training deep neural networks.
- Loss Function: Categorical crossentropy, suitable for multiclass classification tasks.
- Metrics: Accuracy to evaluate the performance of the model.

## Part 1 - Multiclass Classification
Corresponding to the first notebook, the multiclass classification model classifies each image into one of the 10 specific fashion categories.

The multiclasse model obtained a test Accuracy of 0.868 and validation Accuracy similar to test accuracy, indicating good generalization.
This lower performance compared to the binary classification model is due to the complexity of distinguishing between 10 classes, some of which are visually similar.

Then binarization of the multiclasse model outputs was carried out in which a function was build in order to map the outputs into binary categories of "Clothing" and "Footwear and Bags" based on their predicted class.

The accuracy obtained by binarizing the outputs of the multiclass model is slightly lower than that of the standalone binary model. This difference is due to the intermediate step of multiclass prediction, which introduces more opportunities for error.

## Part 2 - Binary Classification
The binary classification model aims to distinguish between "Clothing" and "Footwear and Bags." The labels are mapped as follows:
- Clothing: T-shirt/top, Trouser, Pullover, Dress, Coat, Shirt
- Footwear and bags: Sandal, Sneaker, Bag, Ankle boot

The binary model obtained a test Accuracy of 0.994 and validation Accuracy similar to test accuracy, indicating good generalization. This model performs exceptionally well, achieving an accuracy of 0.994. This high performance can be attributed to the clear distinction between the two classes (Clothing vs. Footwear and Bags), making it easier for the model to differentiate.

## Evaluation
### Performance Comparison
- Multiclass Neural Network: Accuracy = 0.868
- Binary Neural Network: Accuracy = 0.994
- Binary Classification via Multiclass Model: Slightly lower than direct binary classification but higher than multiclass accuracy.

The binary classification model outperforms the multiclass classification model due to the inherent similarities between some clothing items in the multiclass setup. The multiclass model's performance is affected by the difficulty in distinguishing between visually similar classes, leading to lower accuracy compared to the binary model.

## Conclusion
The binary classification model significantly outperforms the multiclass classification model, achieving near-perfect accuracy of 0.994 compared to 0.868. This performance difference highlights the complexity of multiclass classification, especially with visually similar items such as T-shirts, Shirts, Pullovers, and Coats. These similarities lead to frequent misclassifications, as seen in the confusion matrix of the multiclass model.

The primary reason for the binary model's superior performance lies in the simplicity of differentiating between two broad categories: "Clothing" and "Footwear and Bags." This task is inherently easier because the differences between these categories are more pronounced. Conversely, the multiclass model must handle ten detailed classes, leading to more complex decision boundaries and higher susceptibility to class imbalance.

When using the multiclass model's outputs for binary classification, intermediate classification errors reduce overall performance, though it still performs better than the standalone multiclass model.

In summary, while the multiclass model provides detailed categorization, its performance is hindered by class similarity and complexity. The binary classification model, with its broader categories, achieves higher accuracy by simplifying the task. Binarizing the outputs of the multiclass model offers a balanced approach, combining detailed insights with higher accuracy. This analysis underscores the importance of model simplicity and the impact of class similarity and imbalance on classification performance.
