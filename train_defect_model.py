import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from shutil import copyfile, rmtree
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, cohen_kappa_score, jaccard_score
from imblearn.metrics import geometric_mean_score

# Paths
original_dataset = "automation_dataset"
binary_dataset = "binary_defect_dataset"
img_size = (128, 128)
batch_size = 32

# Step 1: Preprocess - Convert to binary classification dataset
def prepare_binary_dataset():
    if os.path.exists(binary_dataset):
        rmtree(binary_dataset)

    os.makedirs(os.path.join(binary_dataset, "defective"), exist_ok=True)
    os.makedirs(os.path.join(binary_dataset, "non_defective"), exist_ok=True)

    for category in os.listdir(original_dataset):
        category_path = os.path.join(original_dataset, category)
        if not os.path.isdir(category_path):
            continue

        target_label = "non_defective" if category == "flawless" else "defective"
        for img in os.listdir(category_path):
            src = os.path.join(category_path, img)
            dst = os.path.join(binary_dataset, target_label, img)
            copyfile(src, dst)

    print("✅ Binary dataset prepared.")

# Step 2: Create data loaders
def get_data_generators():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        binary_dataset,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training"
    )

    val_generator = datagen.flow_from_directory(
        binary_dataset,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation"
    )
    return train_generator, val_generator

# Step 3: Define CNN Model
def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Step 4: Train
def train():
    prepare_binary_dataset()
    train_gen, val_gen = get_data_generators()
    model = build_model()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )

    # Get validation data and predictions
    val_gen.reset()
    y_true = []
    y_pred = []
    y_prob = []
    for i in range(len(val_gen)):
        X, y = val_gen[i]
        preds = model.predict(X).flatten()
        y_prob.extend(preds)
        y_pred.extend((preds > 0.5).astype(int))
        y_true.extend(y.astype(int))
        if (i+1)*val_gen.batch_size >= val_gen.samples:
            break

    # 1. Accuracy vs Loss Curve
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Accuracy and Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('accuracy_loss_curve.png')
    plt.close()

    # 2. Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 4. ROC Curve + AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.close()
    print(f"AUC Score: {auc_score:.4f}")

    # 5. Cohen's Kappa Score
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa Score: {kappa:.4f}")

    # 6. G-Mean Score
    gmean = geometric_mean_score(y_true, y_pred)
    print(f"G-Mean Score: {gmean:.4f}")

    # 7. Mean IoU Score
    iou = jaccard_score(y_true, y_pred, average='binary')
    print(f"Mean IoU Score: {iou:.4f}")

    # Save confusion matrix as image
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    plt.savefig('confusion_matrix.png')
    plt.close()

    model.save("defect_detector_model.h5")
    print("✅ Model saved as defect_detector_model.h5")

if __name__ == "__main__":
    train()
