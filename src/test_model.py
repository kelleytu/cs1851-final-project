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

def apply_tta_batch(images, mode):
    if mode == "orig":
        return images
    elif mode == "hflip":
        return torch.flip(images, dims=[3])
    elif mode == "vflip":
        return torch.flip(images, dims=[2])
    elif mode == "hvflip":
        return torch.flip(torch.flip(images, dims=[3]), dims=[2])
    else:
        raise ValueError(f"Unknown TTA mode: {mode}")

def predict_tta_simple(model, X_img, X_tab):
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

def predict_tta(model, X_img, X_tab, batch_size=64):
    tta_probs = []
    tta_modes = ["orig", "hflip", "vflip", "hvflip"]
    with torch.no_grad():
        for mode in tta_modes:
            aug_img = apply_tta_batch(X_img, mode)
            probs_chunk = []
            n = aug_img.size(0)
            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)
                batch_imgs = aug_img[i:end]
                batch_tab = X_tab[i:end]

                logits = model(batch_imgs, batch_tab)
                probs = torch.softmax(logits, dim=1)
                probs_chunk.append(probs)
            tta_probs.append(torch.cat(probs_chunk, dim=0))
    avg_probs = torch.stack(tta_probs, dim=0).mean(dim=0)
    preds = torch.argmax(avg_probs, dim=1).numpy()
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
# test_tabular = StandardScaler().fit_transform(test_tabular)

# preprocess tabular data
test_tabular = preprocess_tabular_test(test_tabular, save_path="saved_models/15_sched/tabular_preprocessor.pkl")
tabular_dim = test_tabular.shape[1]
print("Processed Tabular Shape:", test_tabular.shape)
test_tabular = torch.tensor(test_tabular).float()

# load in model
res_net_model = ResNetFusionModel(tabular_dim=tabular_dim, num_classes=num_classes)
res_net_model.load_state_dict(torch.load("saved_models/15_sched/resnet_fusion_model.pt", map_location="cpu", weights_only=True))
res_net_model.eval()

# get predictions
with torch.no_grad():
    logits = res_net_model(test_images, test_tabular)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1).cpu().numpy()
# preds = predict_tta_simple(res_net_model, test_images, test_tabular)
# preds = predict_tta(res_net_model, test_images, test_tabular, batch_size=64)

# submission
submission = pd.DataFrame({"ID" : test_ids, "label" : preds})
print(submission.head())
print(submission["label"].value_counts())
print(submission.shape)

submission.to_csv("submission3_test1_test2.csv", index=False)
print("Saved submission file: submission_test1_test2.csv")


# load and evaluate models
# tab_model = joblib.load("tab_model.pkl")
# tabular_probs = tab_model.predict(test_tabular, return_proba = True)
 

# state_dict = torch.load("cnn_weights_no_erosion.pth", weights_only=True)
# cnn_model = BasicCNN(num_classes)
# cnn_model.load_state_dict(state_dict)


# cnn_model.eval()
# with torch.no_grad():
#     logits = cnn_model(test_images)
#     cnn_probs = torch.nn.functional.softmax(logits, dim=1).numpy()



# alpha = 0.6
# combined_probs = alpha * tabular_probs + (1 - alpha) * cnn_probs
# combined_preds = np.argmax(combined_probs, axis = 1)

# np.savetxt('ensemble_preds.csv', combined_preds, delimiter=',') 
