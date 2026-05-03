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
import os


loader = FPDataLoader()
train_ids, train_images, train_labels, train_tabular, metadata = loader.get_cancer_data()
# loader.show_sample_imgs(train_images=train_images, train_labels=train_labels)

X_img_train, X_img_val, X_tab_train, X_tab_val, ids_train, ids_val, y_train, y_val = train_test_split(
    train_images, 
    train_tabular, 
    train_ids, 
    train_labels,
    test_size=0.2,
    random_state=42, 
    stratify=train_labels)

# sample_images = preprocess_images(X_img_train, show=False) 
# sample_images = augment_data(sample_images, show=True)

X_img_train = preprocess_images(X_img_train)
X_img_val = preprocess_images(X_img_val)
X_tab_train, X_tab_val, _ = preprocess_tabular(X_tab_train, X_tab_val)


EPOCHS=20
num_classes=7
tabular_dim = X_tab_train.shape[1]

model_base = ResNetFusionModel(
    tabular_dim=tabular_dim,
    num_classes=num_classes,
    dropout=0.3,
)

history, metrics, model_probs, best_embeddings = model_base.run_experiment(
    X_train_img=X_img_train,
    X_train_tab=X_tab_train,
    X_val_img=X_img_val,
    X_val_tab=X_tab_val,
    y_train=y_train,
    y_val=y_val,
    num_epochs=EPOCHS,
    batch_size=64
)
print(history)
print(pd.Series(metrics))

plot_cm(metrics["confusion_matrix"])

path_name = f"saved_models/{EPOCHS}/"
model_name = f"resnet_fusion_model_{metrics['f1']:.4f}"

os.makedirs(path_name, exist_ok=True)
torch.save(model_base.state_dict(), path_name + model_name + ".pt")
# joblib.dump(history, "resnet_fusion_history.pkl")
joblib.dump(metrics, path_name + model_name + "_metrics.pkl")
print("Saved model weights, history, and metrics.")

history = pd.DataFrame(history)
history.to_csv(path_name + "history.csv", index=False)

torch.save(best_embeddings, path_name + "val_embeddings.pt")