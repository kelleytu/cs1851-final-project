from preprocessing import FPDataLoader, preprocess_images
from model import BasicCNN
from model import GradientBoostingModel
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import joblib


loader = FPDataLoader()
train_ids, train_images, train_labels, train_tabular, metadata = loader.get_cancer_data()
# loader.show_sample_imgs(train_images=train_images, train_labels=train_labels)

train_images = preprocess_images(train_images)

EPOCHS=1
num_classes=7

tabular_model = GradientBoostingModel(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=500,
    subsample=0.8
)
X_img_train, X_img_test, X_tab_train, X_tab_test, ids_train, ids_test, y_train, y_test = train_test_split(
    train_images, 
    train_tabular, 
    train_ids, 
    train_labels,
    test_size=0.2,
    random_state=42, 
    stratify=train_labels)

# tune_dict = tabular_model.tune_hyperparameters(X_tab_train, y_train, param_grid={
#     'max_depth': [1, 2, 3, 4, 5],
#     'learning_rate': [0.01, 0.05, 0.1, 0.15],
#     'n_estimators': [100, 200, 500, 800],
#     'subsample': [0.6, 0.8, 1.0]
# })
# print("\nBest hyperparameters:", tune_dict["best_params"], "\n")

tabular_model.fit(X_tab_train, y_train)
tabular_probs = tabular_model.predict(X_tab_test, return_proba=True)
metrics = tabular_model.evaluate(X_tab_test, y_test)
print(pd.Series(metrics))

print("SAVING TABULAR MODEL")
joblib.dump(tabular_model, "tab_model.pkl")

model_base = BasicCNN(num_classes)
history, metrics, cnn_probs = model_base.run_experiment(
    model_base, 
    X_train=X_img_train, 
    X_test=X_img_test, 
    y_train=y_train, 
    y_test=y_test, 
    num_epochs=EPOCHS)
# probs for ensemble
print(history)
print(pd.Series(metrics))

print("SAVING CNN MODEL")
torch.save(model_base.state_dict(), "cnn_weights_no_erosion.pth")

cnn_probs = cnn_probs.numpy()
np.savetxt('cnn_probs.csv', cnn_probs, delimiter=',') 
cnn_probs = np.loadtxt('cnn_probs.csv', delimiter=',')

alpha = 0.6
combined_probs = alpha * tabular_probs + (1 - alpha) * cnn_probs
combined_preds = np.argmax(combined_probs, axis = 1)


print("\nEnsemble Metrics:")
print("Accuracy:", accuracy_score(y_test, combined_preds))
print("Precision:", precision_score(y_test, combined_preds, zero_division = 0, average = "macro"))
print("Recall:", recall_score(y_test, combined_preds, zero_division = 0, average = "macro"))
print("F1:", f1_score(y_test, combined_preds, zero_division = 0, average = "macro"))
print("ROC-AUC", roc_auc_score(y_test, combined_probs, average = "macro", multi_class = "ovr"))

