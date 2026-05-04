from preprocessing import *
from model import *
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import joblib

num_classes=7
tabular_dim = 3

def predict_tta_simple(model, X_img, X_tab):
    '''
    Predicts class based on averaged model performance on original and augmented images
    '''
    tta_probs = []
    with torch.no_grad():
        test_variants = [
            X_img,
            torch.flip(X_img, dims=[3]),                   
            torch.flip(X_img, dims=[2]),                   
            torch.flip(torch.flip(X_img, dims=[3]), dims=[2]),
        ]

        for _, imgs in enumerate(test_variants):
            logits = model(imgs, X_tab)
            probs = torch.softmax(logits, dim=1)
            tta_probs.append(probs)

        avg_probs = torch.stack(tta_probs, dim=0).mean(dim=0)
        preds = torch.argmax(avg_probs, dim=1).cpu().numpy()
    return preds


# load data
loader = FPDataLoader()
test1_ids, test1_images, test1_tabular = loader.get_test_cancer_data(1)
test2_ids, test2_images, test2_tabular = loader.get_test_cancer_data(2)

# concatenate datasets 1 and 2
test_ids = np.concatenate([test1_ids, test2_ids])
test_images = np.concatenate([test1_images, test2_images])
test_tabular = np.concatenate([test1_tabular, test2_tabular])

# preprocess and use eval transform on images
test_images = preprocess_images(test_images)
test_images = change_img_format(test_images)
test_images = transform_data(test_images, train=False)

# preprocess tabular data
test_tabular = preprocess_tabular_test(test_tabular, save_path="saved_models/20/tabular_preprocessor.pkl")
tabular_dim = test_tabular.shape[1]
print("Processed Tabular Shape:", test_tabular.shape)
test_tabular = torch.tensor(test_tabular).float()

# load in model
res_net_model = ResNetFusionModel(tabular_dim=tabular_dim, num_classes=num_classes)
res_net_model.load_state_dict(torch.load("saved_models/20/resnet_fusion_model_0.6143.pt", map_location="cpu", weights_only=True))
res_net_model.eval()

# get predictions
# with torch.no_grad():
#     logits = res_net_model(test_images, test_tabular)
#     probs = torch.softmax(logits, dim=1)
#     preds = torch.argmax(probs, dim=1).cpu().numpy()
preds = predict_tta_simple(res_net_model, test_images, test_tabular)

# submission
submission = pd.DataFrame({"ID" : test_ids, "label" : preds})
print(submission.head())
print(submission["label"].value_counts())
print(submission.shape)

submission.to_csv("simple_tta_submission.csv", index=False)
print("Saved submission file: submission_test1_test2.csv")