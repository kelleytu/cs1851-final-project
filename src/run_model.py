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
from sklearn.preprocessing import StandardScaler


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

# sample_images = preprocess_images(X_img_train[:3], show=False)
# sample_images = augment_data(sample_images, show=True)

X_img_train = preprocess_images(X_img_train)
X_img_test = preprocess_images(X_img_test)

tab_scaler = StandardScaler()
X_tab_train = tab_scaler.fit_transform(X_tab_train)
X_tab_val = tab_scaler.transform(X_tab_val)
X_tab_test = tab_scaler.transform(X_tab_test)

EPOCHS=20
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

history = pd.DataFrame(history)
history.to_csv("history.csv", index=False)
