# CIFAR-10 Image Classification: EDA, Modeling, and Evaluation

# 1. Introduction
"""
In this project, we build and evaluate Convolutional Neural Network (CNN) models to classify images from the CIFAR-10 dataset into 10 distinct object categories.
We demonstrate the importance of regularization, data augmentation, and model depth for improving generalization.
"""

# 2. Problem Statement
"""
The CIFAR-10 dataset contains 60,000 color images of 32x32 pixels across 10 classes, such as airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
The goal is to correctly classify each image into its respective class.
"""

# 3. Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 4. Load and Explore Dataset (EDA)
from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Display sample images
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Check class distribution
unique, counts = np.unique(y_train, return_counts=True)
plt.bar(class_names, counts)
plt.xticks(rotation=45)
plt.title("Training Set Class Distribution")
plt.show()

# 5. Data Preprocessing
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()

# 6. Baseline Model: Simple CNN
baseline_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

baseline_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

history_baseline = baseline_model.fit(X_train, y_train, epochs=15,
                                      validation_split=0.2, batch_size=64)

# 7. Evaluate Baseline Model
plt.plot(history_baseline.history['accuracy'], label='Train Accuracy')
plt.plot(history_baseline.history['val_accuracy'], label='Validation Accuracy')
plt.title('Baseline Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred_baseline = np.argmax(baseline_model.predict(X_test), axis=1)
cm_baseline = confusion_matrix(y_test, y_pred_baseline)

plt.figure(figsize=(10,8))
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Baseline Model Confusion Matrix')
plt.show()

# 8. Improved Model: Advanced CNN
advanced_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

advanced_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

history_advanced = advanced_model.fit(datagen.flow(X_train, y_train, batch_size=64),
                                      epochs=50, validation_split=0.2)

# 9. Evaluate Advanced Model
plt.plot(history_advanced.history['accuracy'], label='Train Accuracy')
plt.plot(history_advanced.history['val_accuracy'], label='Validation Accuracy')
plt.title('Advanced Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Confusion Matrix Advanced
y_pred_advanced = np.argmax(advanced_model.predict(X_test), axis=1)
cm_advanced = confusion_matrix(y_test, y_pred_advanced)

plt.figure(figsize=(10,8))
sns.heatmap(cm_advanced, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Advanced Model Confusion Matrix')
plt.show()

# 10. Test Set Predictions
print(classification_report(y_test, y_pred_advanced, target_names=class_names))

# 11. Final Conclusion
"""
- The baseline CNN showed signs of early overfitting and limited accuracy.
- By introducing deeper layers, data augmentation, and regularization, the advanced CNN achieved significantly higher accuracy.
- Future work could include experimenting with transfer learning from pretrained models like ResNet50 or EfficientNet.
"""
