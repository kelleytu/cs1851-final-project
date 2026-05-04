import pickle
import matplotlib.pyplot as plt
import pandas as pd
import sys 
from utils import *
from preprocessing import *
import torch
from model import *
import seaborn as sns

def plot_model_loss(csv_path):
    # with open(pkl_path, 'rb') as file:
    #     history = pickle.load(file)
    history = pd.read_csv(csv_path)
    metrics_list = ["val_loss", "val_acc", "val_f1"]

    fig, axes = plt.subplots(1,3, figsize=(15,4))
    num_epochs = len(history["val_loss"])

    for i, m in enumerate(metrics_list):
        axes[i].plot(history[m])
        axes[i].set_title(f"{m} over {num_epochs} epochs")
        axes[i].set_xlabel("epoch")
        axes[i].set_ylabel(m)

    plt.tight_layout()
    plt.show()

def plot_cm(cm):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix of Best Model")
    plt.show()


path = sys.argv[1]

csv_path = path + "/history.csv"
plot_model_loss(csv_path)


# gets and plots embeddings of best model 
embedding_path = path + "/val_embeddings.pt"
embds = torch.load(embedding_path)

cmb = embds["combined"].numpy()
img = embds["img"].numpy()
tab = embds["tab"].numpy()
labels = embds["labels"].numpy()

plot_embedding(cmb, labels, "combined", random_state=42)
plot_embedding(img, labels, "image", random_state=42)
plot_embedding(tab, labels, "tabular", random_state=42)


loader = FPDataLoader()
train_ids, train_images, train_labels, train_tabular, metadata = loader.get_cancer_data()

X_img_train, X_img_val, X_tab_train, X_tab_val, ids_train, ids_val, y_train, y_val = train_test_split(
    train_images, 
    train_tabular, 
    train_ids, 
    train_labels,
    test_size=0.2,
    random_state=42, 
    stratify=train_labels)

img_index = 32

# preprocess and use eval transform on images
preprocess_index = img_index + 2
gcam_img = preprocess_images(X_img_train[:preprocess_index])
gcam_img = change_img_format(gcam_img)
gcam_img = transform_data(gcam_img, train=False)

# preprocess tabular data
gcam_tab = preprocess_tabular_test(X_tab_train, save_path="saved_models/15_sched/tabular_preprocessor.pkl")
tabular_dim = gcam_tab.shape[1]
gcam_tab = torch.tensor(gcam_tab).float()

# load in model
res_net_model = ResNetFusionModel(tabular_dim=tabular_dim, num_classes=7)
res_net_model.load_state_dict(torch.load("saved_models/15_sched/resnet_fusion_model.pt", map_location="cpu", weights_only=True))
res_net_model.eval()

grad_cam(res_net_model, gcam_img[img_index].unsqueeze(0), gcam_tab[img_index].unsqueeze(0), X_img_train[img_index], y_train[img_index])