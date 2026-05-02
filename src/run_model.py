from preprocessing import *
from model import *
from utils import *
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

X_img_train, X_img_test, X_tab_train, X_tab_test, ids_train, ids_test, y_train, y_test = train_test_split(
    train_images, 
    train_tabular, 
    train_ids, 
    train_labels,
    test_size=0.2,
    random_state=42, 
    stratify=train_labels)

# train_images = preprocess_images(train_images)
X_img_train = preprocess_images(X_img_train)

X_img_test = preprocess_images(X_img_test)
X_img_train = normalize_for_resnet(X_img_train)
X_img_test = normalize_for_resnet(X_img_test)

EPOCHS=40
num_classes=7
tabular_dim = X_tab_train.shape[1]

model_base = ResNetFusionModel(
    tabular_dim=tabular_dim,
    num_classes=num_classes,
    dropout=0.3,
)

history, metrics, model_probs = model_base.run_experiment(
    X_train_img=X_img_train,
    X_train_tab=X_tab_train,
    X_test_img=X_img_test,
    X_test_tab=X_tab_test,
    y_train=y_train,
    y_test=y_test,
    num_epochs=EPOCHS,
    batch_size=64
)
print(history)
print(pd.Series(metrics))

torch.save(model_base.state_dict(), "resnet_fusion_model.pt")
joblib.dump(history, "resnet_fusion_history.pkl")
joblib.dump(metrics, "resnet_fusion_metrics.pkl")
print("Saved model weights, history, and metrics.")




# fusion_model = FusionModel(num_classes)

# history, metrics, model_probs = fusion_model.run_experiment(
#     model=fusion_model,
#     X_train_img=X_img_train,
#     X_train_tab=X_tab_train,
#     X_test_img=X_img_test,
#     X_test_tab=X_tab_test,
#     y_train=y_train,
#     y_test=y_test,
#     num_epochs=EPOCHS
# )
# print(history)
# print(pd.Series(metrics))




# model_base = ImageClassifier(
#     num_classes=num_classes,
#     dropout=0.3,
# )

# history, metrics, model_probs = model_base.run_experiment(
#     X_train=X_img_train,
#     X_test=X_img_test,
#     y_train=y_train,
#     y_test=y_test,
#     num_epochs=EPOCHS
# )
# print(history)
# print(pd.Series(metrics))


# print("\nEnsemble Metrics:")
# print("Accuracy:", accuracy_score(y_test, combined_preds))
# print("Precision:", precision_score(y_test, combined_preds, zero_division = 0, average = "macro"))
# print("Recall:", recall_score(y_test, combined_preds, zero_division = 0, average = "macro"))
# print("F1:", f1_score(y_test, combined_preds, zero_division = 0, average = "macro"))
# print("ROC-AUC", roc_auc_score(y_test, combined_probs, average = "macro", multi_class = "ovr"))

