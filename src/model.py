import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from preprocessing import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils import FusionDataset
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


class FusionModel(nn.Module):
    def __init__(self, num_classes=7, dropout_p=0.3):
        super().__init__()
        self.loader = FPDataLoader()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Dropout2d(p=0.25),
            nn.Flatten(),
        )
        self.tabular_head = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(802880, 512), 
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_image, x_tabular):
        # return self.classifier(self.cnn(x))
        cnn_out = self.cnn(x_image)
        tabular_out = self.tabular_head(x_tabular)
        fused = torch.cat([cnn_out, tabular_out], dim=1)
        return self.classifier(fused)

    def train_one_epoch(self, model, X_img, X_tab, y, optimizer, criterion, batch_size):

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
        total_loss = 0.0
        prob_list = []

        for i, (X_img, X_tab, y) in enumerate(train_loader):

            print(f"batch {i}")
            model.train()
            optimizer.zero_grad()
            logits = model(X_img, X_tab)

            # probs = torch.nn.functional.softmax(logits, dim=1)
            # prob_list.append(probs.cpu())

            loss = criterion(logits, y)

            # print("backward pass")
            loss.backward()
            # print("optimize step")
            optimizer.step()

            total_loss += loss.item()

        # prob_list = torch.cat(prob_list, dim=0)
        # return total_loss / len(train_loader), probs
        return total_loss / len(train_loader)


    def evaluate(self, model, X_img, X_tab, y, criterion):
        model.eval()
        with torch.no_grad():


            logits = model(X_img, X_tab)
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
    
    def run_experiment(self, model, X_train_img, X_train_tab, X_test_img, X_test_tab, y_train, y_test, num_epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        history = {"train_loss": [], "test_loss": [], "test_acc": []}

        # later change to use preprocessing method?
        X_train_img = np.transpose(X_train_img, (0, 3, 1, 2)) # transposed images here
        X_test_img = np.transpose(X_test_img, (0, 3, 1, 2)) # transposed images here

        X_train_img = torch.tensor(X_train_img).float()
        X_test_img = torch.tensor(X_test_img).float()
        X_train_tab = torch.tensor(X_train_tab).float()
        X_test_tab = torch.tensor(X_test_tab).float()


        y_train = torch.tensor(y_train).long()
        y_test = torch.tensor(y_test).long()

  
        for epoch in range(1, num_epochs + 1):
            tr_loss = model.train_one_epoch(model, X_train_img, X_train_tab, y_train, optimizer, criterion, batch_size=64) # probs for ensemble model
            te_loss, te_acc, metrics, probs = model.evaluate(model, X_test_img, X_test_tab, y_test, criterion)
            history["train_loss"].append(tr_loss)
            history["test_loss"].append(te_loss)
            history["test_acc"].append(te_acc)
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | train loss {tr_loss:.4f} | "
                    f"test loss {te_loss:.4f} | test acc {te_acc:.3f}")

        return history, metrics, probs

# may need to run open "/Applications/Python 3.12/Install Certificates.command" in terminal
# from torchvision import models
# class ImageClassifier(nn.Module):
#     def __init__(self, num_classes=7, dropout=0.3):
#         super().__init__()
#         res_mod = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         in_features = res_mod.fc.in_features
#         res_mod.fc = nn.Sequential(
#             nn.Dropout(p=dropout),
#             nn.Linear(in_features, num_classes)
#         )
#         self.image_classifier = res_mod

#     def forward(self, x):
#         return self.image_classifier(x)

#     def train_one_epoch(self, X_img, y, optimizer, criterion, batch_size):
#         perm = torch.randperm(X_img.size(0))
#         X_img = X_img[perm]
#         y = y[perm]
        
#         train_loader = DataLoader(
#             TensorDataset(X_img, y),
#             batch_size=batch_size,
#             shuffle=False
#         )

#         # num_batches = (.shape[0] + batch_size - 1) // batch_size
#         self.train()
#         total_loss = 0.0

#         for i, (X_img, y) in enumerate(train_loader):
#             print(f"batch {i}")
#             optimizer.zero_grad()
#             logits = self(X_img)

#             # probs = torch.nn.functional.softmax(logits, dim=1)
#             # prob_list.append(probs.cpu())
#             loss = criterion(logits, y)

#             # print("backward pass")
#             loss.backward()
#             # print("optimize step")
#             optimizer.step()
#             total_loss += loss.item()

#         # prob_list = torch.cat(prob_list, dim=0)

#         # print(prob_list.size())
#         # flatten if needed. should be size , num_classes

#         # return total_loss / len(train_loader), probs
#         return total_loss / len(train_loader)


#     def evaluate(self, X_img, y, criterion):
#         self.eval()
#         with torch.no_grad():
#             logits = self(X_img)
#             y_proba = torch.nn.functional.softmax(logits, dim=1)

#             loss = criterion(logits, y)
#             y_pred = logits.argmax(dim=1)
#             correct = (y_pred == y).sum().item()
#             n = X_img.shape[0]

#         avg = "macro"
#         metrics = {
#             "accuracy": accuracy_score(y, y_pred),
#             "precision": precision_score(y, y_pred, zero_division = 0, average = avg),
#             "recall": recall_score(y, y_pred, zero_division = 0, average = avg),
#             "f1": f1_score(y, y_pred, zero_division = 0, average = avg),
#             "roc_auc": np.nan
#         }
#         metrics["roc_auc"] = roc_auc_score(y, y_proba, average = avg, multi_class = "ovr")

#         return loss.item(), correct / n, metrics, y_proba
    
#     def run_experiment(self, X_train, X_test, y_train, y_test, num_epochs):
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         history = {"train_loss": [], "test_loss": [], "test_acc": []}

#         # later change to use preprocessing method?
#         X_train = np.transpose(X_train, (0, 3, 1, 2)) # transposed images here
#         X_test = np.transpose(X_test, (0, 3, 1, 2)) # transposed images here

#         X_train = torch.tensor(X_train).float()
#         X_test = torch.tensor(X_test).float()
#         y_train = torch.tensor(y_train).long()
#         y_test = torch.tensor(y_test).long()

#         for epoch in range(1, num_epochs + 1):
#             tr_loss = self.train_one_epoch(X_train, y_train, optimizer, criterion, batch_size=64) # probs for ensemble model
#             te_loss, te_acc, metrics, probs = self.evaluate(X_test, y_test, criterion)
#             history["train_loss"].append(tr_loss)
#             history["test_loss"].append(te_loss)
#             history["test_acc"].append(te_acc)
#             if epoch % 10 == 0:
#                 print(f"  Epoch {epoch:3d} | train loss {tr_loss:.4f} | "
#                     f"test loss {te_loss:.4f} | test acc {te_acc:.3f}")

#         return history, metrics, probs

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

            # probs = torch.nn.functional.softmax(logits, dim=1)
            # prob_list.append(probs.cpu())
            loss = criterion(logits, y)

            # print("backward pass")
            loss.backward()
            # print("optimize step")
            optimizer.step()
            total_loss += loss.item()

        # prob_list = torch.cat(prob_list, dim=0)

        # print(prob_list.size())
        # flatten if needed. should be size , num_classes

        # return total_loss / len(train_loader), probs
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
    
    def run_experiment(self, X_train_img, X_train_tab, X_test_img, X_test_tab, y_train, y_test, num_epochs, batch_size=64):
        history = {"train_loss": [], "test_loss": [], "test_acc": [], "test_f1": [], "epoch": []}

        X_train_img = np.transpose(X_train_img, (0, 3, 1, 2)) # transposed images here
        X_test_img = np.transpose(X_test_img, (0, 3, 1, 2)) # transposed images here

        X_train_img = torch.tensor(X_train_img).float()
        X_test_img = torch.tensor(X_test_img).float()
        X_train_tab = torch.tensor(X_train_tab).float()
        X_test_tab = torch.tensor(X_test_tab).float()

        y_train = torch.tensor(y_train).long()
        y_test = torch.tensor(y_test).long()

        # assign higher priority to minority data
        num_classes = len(torch.unique(y_train))
        class_weights = torch.tensor(compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_train.cpu().numpy()), dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(
            [
                {"params": self.image_classifier.parameters(), "lr": 1e-5},
                {"params": self.tabular_head.parameters(), "lr": 1e-4},
                {"params": self.classifier.parameters(), "lr": 1e-4},
            ], 
            weight_decay=1e-4)
        
        best_f1 = -1
        best_state = None
        best_metrics = None
        best_probs = None

        for epoch in range(1, num_epochs + 1):
            X_train_img = augment_data(X_train_img)

            tr_loss = self.train_one_epoch(X_train_img, X_train_tab, y_train, optimizer, criterion, batch_size=batch_size) # probs for ensemble model
            te_loss, te_acc, metrics, probs = self.evaluate(X_test_img, X_test_tab, y_test, criterion)
            history["train_loss"].append(tr_loss)
            history["test_loss"].append(te_loss)
            history["test_acc"].append(te_acc)
            history["test_f1"].append(metrics["f1"])
            history["epoch"].append(epoch)

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                # getting frozen copy of state dict
                best_state = copy.deepcopy(self.state_dict())
                best_metrics = metrics
                best_probs = probs

            # if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train loss {tr_loss:.4f} | "
                f"test loss {te_loss:.4f} | test acc {te_acc:.4f} | test f1 {metrics['f1']:.4f}")

        self.load_state_dict(best_state)
        return history, best_metrics, best_probs