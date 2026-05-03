import pickle
import matplotlib.pyplot as plt
import pandas as pd
import sys 
from utils import *
from preprocessing import *
import torch
from model import *

def plot_model_loss(csv_path):
    # with open(pkl_path, 'rb') as file:
    #     history = pickle.load(file)
    history = pd.read_csv(csv_path)

    # print(history)
    # train loss, test loss test acc, f1
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



# pkl_path = 'saved_models/20_aug/resnet_fusion_history.pkl'
# csv_path = 'saved_models/5_border/history.csv'

# path = sys.argv[1]

# csv_path = path + "/history.csv"
# # plot_model_loss(csv_path)


# embedding_path = path + "/val_embeddings.pt"
# embds = torch.load(embedding_path)

# cmb = embds["combined"].numpy()
# img = embds["img"].numpy()
# tab = embds["tab"].numpy()
# labels = embds["labels"].numpy()

# plot_embedding(cmb, labels, "combined", random_state=42)
# plot_embedding(img, labels, "image", random_state=42)
# plot_embedding(tab, labels, "tabular", random_state=42)



loader = FPDataLoader()
train_ids, train_images, train_labels, train_tabular, metadata = loader.get_cancer_data()

# preprocess and use eval transform on images
gcam_img = preprocess_images(train_images[:2])
gcam_img = change_img_format(gcam_img)
gcam_img = transform_data(gcam_img, train=False)

# preprocess tabular data
gcam_tab = preprocess_tabular_test(train_tabular, save_path="saved_models/15_sched/tabular_preprocessor.pkl")
tabular_dim = gcam_tab.shape[1]
gcam_tab = torch.tensor(gcam_tab).float()

# load in model
res_net_model = ResNetFusionModel(tabular_dim=tabular_dim, num_classes=7)
res_net_model.load_state_dict(torch.load("saved_models/15_sched/resnet_fusion_model.pt", map_location="cpu", weights_only=True))
res_net_model.eval()

grad_cam(res_net_model, gcam_img[0].unsqueeze(0), gcam_tab[0].unsqueeze(0), train_images[0])