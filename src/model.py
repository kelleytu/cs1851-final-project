import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from preprocessing import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils import *
import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
from sklearn.utils.class_weight import compute_sample_weight

from torchvision import models
from sklearn.utils.class_weight import compute_class_weight
import copy

class ResNetFusionModel(nn.Module):
    def __init__(self, tabular_dim=3, num_classes=7, dropout=0.3):
        super().__init__()
        res_mod = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = res_mod.fc.in_features
        # remove classifier for fusion
        res_mod.fc = nn.Identity()
        self.image_classifier = res_mod
        self.tabular_head = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features+64, 512),  # might need to change first argument
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_image, x_tabular):
        resnet_out = self.image_classifier(x_image)
        tabular_out = self.tabular_head(x_tabular)
        fused = torch.cat([resnet_out, tabular_out], dim=1)
        return self.classifier(fused)

    def train_one_epoch(self, X_img, X_tab, y, optimizer, criterion, batch_size):
        perm = torch.randperm(X_img.size(0))
        X_img = X_img[perm]
        X_tab = X_tab[perm]
        y = y[perm]
        
        train_loader = DataLoader(
            FusionDataset(X_img, X_tab, y),
            batch_size=batch_size,
            shuffle=False
        )

        # num_batches = (.shape[0] + batch_size - 1) // batch_size
        self.train()
        total_loss = 0.0

        for i, (X_img, X_tab, y) in enumerate(train_loader):
            print(f"batch {i}")
            optimizer.zero_grad()
            logits = self(X_img, X_tab)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)


    def evaluate(self, X_img, X_tab, y, criterion):
        self.eval()
        with torch.no_grad():
            logits = self(X_img, X_tab)
            y_proba = torch.nn.functional.softmax(logits, dim=1)

            loss = criterion(logits, y)
            y_pred = logits.argmax(dim=1)
            correct = (y_pred == y).sum().item()
            n = X_img.shape[0]

        avg = "macro"
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division = 0, average = avg),
            "recall": recall_score(y, y_pred, zero_division = 0, average = avg),
            "f1": f1_score(y, y_pred, zero_division = 0, average = avg),
            "roc_auc": np.nan
        }
        metrics["roc_auc"] = roc_auc_score(y, y_proba, average = avg, multi_class = "ovr")

        return loss.item(), correct / n, metrics, y_proba
    
    def run_experiment(self, X_train_img, X_train_tab, X_val_img, X_val_tab, y_train, y_val, num_epochs, batch_size=64):
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "epoch": []}

        X_train_img = change_img_format(X_train_img)
        X_val_img = change_img_format(X_val_img)
        
        X_train_tab = torch.tensor(X_train_tab).float()
        X_val_tab = torch.tensor(X_val_tab).float()

        y_train = torch.tensor(y_train).long()
        y_val = torch.tensor(y_val).long()

        X_val_img = transform_data(X_val_img, train=False)
        # assign higher priority to minority data
        num_classes = len(torch.unique(y_train))
        class_weights = torch.tensor(compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_train.cpu().numpy()), dtype=torch.float32)
        # criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion = FocalLoss(weights=class_weights, gamma=2.0)
        optimizer = optim.AdamW(
            [
                {"params": self.image_classifier.parameters(), "lr": 1e-5},
                {"params": self.tabular_head.parameters(), "lr": 1e-4},
                {"params": self.classifier.parameters(), "lr": 1e-4},
            ], 
            weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, threshold=1e-3, cooldown=1,
            min_lr=1e-6
        )        

        best_f1 = -1
        best_state = None
        best_metrics = None
        best_probs = None

        for epoch in range(1, num_epochs + 1):
            X_train_img_epoch = X_train_img.clone()
            X_train_img_epoch = transform_data(X_train_img_epoch, train=True)

            tr_loss = self.train_one_epoch(X_train_img_epoch, X_train_tab, y_train, optimizer, criterion, batch_size=batch_size) # probs for ensemble model
            val_loss, val_acc, metrics, probs = self.evaluate(X_val_img, X_val_tab, y_val, criterion)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(metrics["f1"])
            history["epoch"].append(epoch)

            scheduler.step(metrics["f1"])

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                # getting frozen copy of state dict
                best_state = copy.deepcopy(self.state_dict())
                best_metrics = metrics
                best_probs = probs

            print(f"  Epoch {epoch:3d} | train loss {tr_loss:.4f} | "
                f"val loss {val_loss:.4f} | val acc {val_acc:.4f} | val f1 {metrics['f1']:.4f}")

        self.load_state_dict(best_state)
        return history, best_metrics, best_probs