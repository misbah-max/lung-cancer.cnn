# -------------------------------
# STEP 1: IMPORTS
# -------------------------------
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import itertools

# -------------------------------
# STEP 2: ATTACH DATASET
# -------------------------------
# Make sure to add the dataset via "Add Data" in Kaggle notebook sidebar
base_path = '/kaggle/input/lung-and-colon-cancer-histopathological-images/lung_colon_image_set/lung_image_sets'

# Check folders
print("Dataset folders:", os.listdir(base_path))

# -------------------------------
# STEP 3: CREATE DATAFRAME
# -------------------------------
filepaths = []
labels = []

folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    images = glob.glob(os.path.join(folder_path, '*.png')) + \
             glob.glob(os.path.join(folder_path, '*.jpg')) + \
             glob.glob(os.path.join(folder_path, '*.jpeg'))
    for img in images:
        filepaths.append(img)
        labels.append(folder)

df = pd.DataFrame({'filepaths': filepaths, 'labels': labels})
print("Total images:", len(df))
print(df['labels'].value_counts())

# Plot class distribution
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='labels')
plt.title('Class Distribution')
plt.show()

# -------------------------------
# STEP 4: TRAIN-VALID-TEST SPLIT
# -------------------------------
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['labels'], random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['labels'], random_state=42)

print(f'Train: {len(train_df)}, Validation: {len(valid_df)}, Test: {len(test_df)}')

# -------------------------------
# STEP 5: IMAGE DATA GENERATORS
# -------------------------------
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepaths',
    y_col='labels',
    target_size=(244,244),
    class_mode='categorical',
    batch_size=32,
    shuffle=True
)

val_data = val_gen.flow_from_dataframe(
    dataframe=valid_df,
    x_col='filepaths',
    y_col='labels',
    target_size=(244,244),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

test_data = test_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filepaths',
    y_col='labels',
    target_size=(244,244),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# -------------------------------
# STEP 6: HANDLE CLASS IMBALANCE
# -------------------------------
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# -------------------------------
# STEP 7: BUILD CNN MODEL
# -------------------------------
def build_model(input_shape=(244,244,3), num_classes=3):
    model = Sequential()
    # Conv Blocks
    model.add(Conv2D(32, (3,3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    # Global Pooling
    model.add(GlobalAveragePooling2D())

    # Dense layers
    for units in [1024, 512, 256]:
        model.add(Dense(units))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    return model

num_classes = df['labels'].nunique()
model = build_model(input_shape=(244,244,3), num_classes=num_classes)

opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# STEP 8: TRAIN MODEL WITH CLASS WEIGHTS
# -------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weights,
    verbose=1
)

# -------------------------------
# STEP 9: PLOT TRAINING METRICS
# -------------------------------
plt.figure(figsize=(12,5))
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# -------------------------------
# STEP 10: EVALUATE ON TEST SET
# -------------------------------
y_pred_prob = model.predict(test_data)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = test_data.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_data.class_indices.keys(),
            yticklabels=test_data.class_indices.keys(), cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))

# -------------------------------
# STEP 11: ROC CURVE
# -------------------------------
y_true_bin = label_binarize(y_true, classes=range(num_classes))

plt.figure(figsize=(8,6))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_pred_prob[:,i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {list(test_data.class_indices.keys())[i]} (AUC = {roc_auc:.2f})')

plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-Class')
plt.legend()
plt.show()
orceProdHost ? "https://www.kaggle.com" : ""}/static/mathjax/2.7.9/MathJax.js?config=TeX-AMS_SVG`;
    head.appendChild(lib);
  });
</script>








  </div>
</body>
</html>

