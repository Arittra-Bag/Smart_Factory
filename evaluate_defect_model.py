import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, cohen_kappa_score, jaccard_score
from imblearn.metrics import geometric_mean_score

# Paths and parameters
binary_dataset = "binary_defect_dataset"
img_size = (128, 128)
batch_size = 32

# Load model
model = load_model("defect_detector_model.h5")

# Data generator for validation/test
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_gen = datagen.flow_from_directory(
    binary_dataset,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# Get all validation data and predictions
y_true = []
y_pred = []
y_prob = []
val_gen.reset()
for i in range(len(val_gen)):
    X, y = val_gen[i]
    preds = model.predict(X).flatten()
    y_prob.extend(preds)
    y_pred.extend((preds > 0.5).astype(int))
    y_true.extend(y.astype(int))
    if (i+1)*val_gen.batch_size >= val_gen.samples:
        break
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

# 1. Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred))

# 2. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)
plt.figure()
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.savefig('confusion_matrix.png')
plt.close()

# 3. ROC Curve + AUC
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

# 4. Cohen's Kappa Score
kappa = cohen_kappa_score(y_true, y_pred)
print(f"Cohen's Kappa Score: {kappa:.4f}")

# 5. G-Mean Score
gmean = geometric_mean_score(y_true, y_pred)
print(f"G-Mean Score: {gmean:.4f}")

# 6. Mean IoU Score
iou = jaccard_score(y_true, y_pred, average='binary')
print(f"Mean IoU Score: {iou:.4f}")

# 7. Accuracy vs Loss Curve (if you have history)
# If you saved the training history (e.g., as a .npy or .pkl file), you can plot it here.
# Otherwise, this is only available during training.

print("All metrics and plots have been generated and saved.")