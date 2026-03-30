import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from preprocessing import FPDataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset



class BasicCNN(nn.Module):
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
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 56 * 56, 512),
            # nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.cnn(x))



    def train_one_epoch(self, model, X_train, y_train, optimizer, criterion):


        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=64,
            shuffle=True
        )
        total_loss = 0.0

        for i, (X, y) in enumerate(train_loader):

            print(f"batch {i}")
            # print("training")
            model.train()


            # print("compute loss")
            optimizer.zero_grad()
            loss = criterion(model(X), y)

            # print("backward pass")
            loss.backward()
            # print("optimize step")
            optimizer.step()

            total_loss += loss.item()


        return total_loss / len(train_loader)



    def evaluate(self, model, X_test, y_test, criterion):
        model.eval()
        with torch.no_grad():

            output = model(X_test)

            loss = criterion(output, y_test)
            correct = (output.argmax(dim=1) == y_test).sum().item()

            n = X_test.shape[0]
        return loss.item(), correct / n
    
    def run_experiment(self, model, X, y, num_epochs):
        criterion = nn.CrossEntropyLoss() #this loss converts the real values into probabilites in it first
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        history = {"train_loss": [], "test_loss": [], "test_acc": []}

        X = np.transpose(X, (0, 3, 1, 2))
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)   

        X_train = torch.tensor(X_train).float()
        X_test = torch.tensor(X_test).float()
        y_train = torch.tensor(y_train).long()
        y_test = torch.tensor(y_test).long()

        # full_dataset = TensorDataset(X, y)
        # train_size = int(0.8 * len(full_dataset)) # change train test split here
        # test_size = len(full_dataset) - train_size
        # train_dataset, test_dataset = random_split(TensorDataset(X, y), [train_size, test_size], generator=torch.Generator().manual_seed(42))     

        for epoch in range(1, num_epochs + 1):
            tr_loss = model.train_one_epoch(model, X_train, y_train, optimizer, criterion)
            te_loss, te_acc = model.evaluate(model, X_test, y_test, criterion)
            history["train_loss"].append(tr_loss)
            history["test_loss"].append(te_loss)
            history["test_acc"].append(te_acc)
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | train loss {tr_loss:.4f} | "
                    f"test loss {te_loss:.4f} | test acc {te_acc:.3f}")
        return history

    def plot_results(self, model, num_epochs):
        epochs  = range(1, num_epochs + 1)

        colors = ["#e74c3c", "#2ecc71"]

        # Loss curves
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.plot(epochs, model_results["train_loss"], label="Train Loss", color=colors[0], linestyle="--", alpha=0.5)
        ax.plot(epochs, hist_base_results["test_loss"],  label="Test Loss", color=colors[1], linestyle="-")
        ax.set_title("Loss (— test,  -- train)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.show()



import matplotlib.pyplot as plt
# import seaborn as sns
from typing import Tuple, Dict, Optional, Union

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


class GradientBoostingModel:
    def __init__(
        self,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        n_estimators: int = 50,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: int = 42,
    ):

        self.params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
        }

        self.model = None
        self.feature_names = None

    def make_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        return train_test_split(X, y, test_size = test_size, random_state = random_state, stratify=y)


    def fit(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True):
        model = GradientBoostingClassifier(**self.params)
        model.fit(X_train, y_train)
        self.model = model

    def predict(
        self, X: np.ndarray, return_proba: bool = False
    ) -> Union[np.ndarray, np.ndarray]:
        # if fitting based on training then transforming on test data
        if return_proba:
            preds = self.model.predict_proba(X)
        else:
            preds = self.model.predict(X)
        return preds

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        y_pred = self.predict(X_test, return_proba = False)

        if len(np.unique(y_test)) == 2:
            avg = "binary"
            y_proba = self.predict(X_test, return_proba = True)[:,1]
        else:
            avg = "macro"
            y_proba = self.predict(X_test, return_proba = True)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division = 0, average = avg),
            "recall": recall_score(y_test, y_pred, zero_division = 0, average = avg),
            "f1": f1_score(y_test, y_pred, zero_division = 0, average = avg),
            "roc_auc": np.nan
        }
        if len(np.unique(y_test)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        else:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba, average = avg, multi_class = "ovr")
        return metrics

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
    ) -> Dict:
        model = GradientBoostingClassifier(**self.params)
        pipeline = model

        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc_ovr"]
        
        results = {}
        for metric in scoring:
            score = cross_val_score(pipeline, X, y, cv=cv, scoring=metric)
            results[metric] = {"mean" : np.mean(score, axis=0), 
                               "std" : np.std(score, axis=0)}
        return results       

    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict,
        cv: int = 3,
        scoring: str = "roc_auc_ovr",
    ) -> Dict:
        pipeline = GradientBoostingClassifier(**self.params)
        grid_search = GridSearchCV(pipeline, param_grid = param_grid, scoring = scoring, cv = cv)
        grid_search.fit(X, y)
        pipeline = grid_search.best_estimator_

        return { "best_params" : grid_search.best_params_,
                "best_score" : grid_search.best_score_,
                "cv_results" : grid_search.cv_results_
                }
        
    # def plot_tree(
    #     self, tree_index: int = 0, figsize: Tuple[int, int] = (20, 15)
    # ) -> None:
    #     tree = self.model.estimators_[tree_index]
    #     plt.figure(figsize = figsize)
    #     plot_tree(tree, feature_names = self.feature_names, filled = True)
    #     plt.title(f"Tree {tree_index} from Gradient Boosting Ensemble")
    #     plt.show()
