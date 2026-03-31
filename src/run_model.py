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
import pandas as pd



loader = FPDataLoader()
train_ids, train_images, train_labels, train_tabular, metadata = loader.get_cancer_data()
# loader.show_sample_imgs(train_images=train_images, train_labels=train_labels)

# print(train_ids[:5])

EPOCHS=1
num_classes=7

# tabular_model = GradientBoostingModel()
# X_train, X_test, y_train, y_test = tabular_model.make_train_test_split(train_tabular, train_labels)

# tune_dict = tabular_model.tune_hyperparameters(X_train, y_train, param_grid={
#     'max_depth': [1, 2, 3, 4],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'n_estimators': [100, 200, 500, 800],
#     'subsample': [0.6, 0.8, 1.0]
# })
# print("\nBest hyperparameters:", tune_dict["best_params"], "\n")

# tabular_model.fit(X_train, y_train)
# tabular_probas = tabular_model.predict(X_test, return_proba=True)
# metrics = tabular_model.evaluate(X_test, y_test)
# print(pd.Series(metrics))

model_base = BasicCNN(num_classes)
history, metrics = model_base.run_experiment(model_base, train_images, train_labels, EPOCHS)
print(history)
print(pd.Series(metrics))



# cat_model = CatBoostModel()
# X_train, X_test, y_train, y_test = cat_model.make_train_test_split(train_tabular, train_labels)
# tune_cat = cat_model.tune_hyperparameters(X_train, y_train, param_grid = {
#     'depth' : [1, 2, 3, 4],
#     'learning_rate' : [0.01, 0.05, 0.1],
#     'n_estimators': [100, 300, 500],
#     'l2_leaf_reg' : [1.0, 3.0, 7.0]
# }, cv = 3, scoring = "f1_macro")
# print("\nBest hyperparameters:", tune_cat["best_params"], "\n")
# cat_model.fit(X_train, y_train)
# cat_probas = cat_model.predict(X_test, return_proba=True)
# metrics_cat = cat_model.evaluate(X_test, y_test)
# print(pd.Series(metrics_cat))

