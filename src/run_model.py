from preprocessing import FPDataLoader
from model import BasicCNN
from model import GradientBoostingModel
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



loader = FPDataLoader()
train_ids, train_images, train_labels, train_tabular, metadata = loader.get_cancer_data()
# loader.show_sample_imgs(train_images=train_images, train_labels=train_labels)

EPOCHS=1
num_classes=7

tabular_model = GradientBoostingModel(
    max_depth=1, 
    learning_rate=0.01,
    n_estimators=200,
    subsample=0.6,
)
X_train, X_test, y_train, y_test = tabular_model.make_train_test_split(train_tabular, train_labels)
tabular_model.fit(X_train, y_train)
metrics = tabular_model.evaluate(X_test, y_test)
print(metrics)

# tune_dict = tabular_model.tune_hyperparameters(train_tabular, train_labels, param_grid={
#     'max_depth': [1, 3, 5],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'n_estimators': [50, 100, 200],
#     'subsample': [0.6, 0.8, 1.0]
# })
# print("Best hyperparameters:", tune_dict["best_params"])



model_base = BasicCNN(num_classes)
hist_base = model_base.run_experiment(model_base, train_images, train_labels, EPOCHS)
hist_base.plot_results(hist_base, EPOCHS)
