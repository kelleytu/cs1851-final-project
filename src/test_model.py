from preprocessing import *
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

num_classes=7

loader = FPDataLoader()
test_ids, test_images, test_tabular, metadata = loader.get_test_cancer_data()

test_images = preprocess_images(test_images)
test_images = change_img_format(test_images)

# load and evaluate models
tab_model = joblib.load("tab_model.pkl")
tabular_probs = tab_model.predict(test_tabular, return_proba = True)
 

state_dict = torch.load("cnn_weights.pth", weights_only=True)
cnn_model = BasicCNN(num_classes)
cnn_model.load_state_dict(state_dict)


cnn_model.eval()
with torch.no_grad():
    logits = cnn_model(test_images)
    cnn_probs = torch.nn.functional.softmax(logits, dim=1).numpy()



alpha = 0.6
combined_probs = alpha * tabular_probs + (1 - alpha) * cnn_probs
combined_preds = np.argmax(combined_probs, axis = 1)

np.savetxt('ensemble_preds.csv', combined_preds, delimiter=',') 
