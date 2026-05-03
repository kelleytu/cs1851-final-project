import pickle
import matplotlib.pyplot as plt
import pandas as pd

def plot_model_loss(pkl_path, csv_path):
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


pkl_path = 'saved_models/20_aug/resnet_fusion_history.pkl'
csv_path = 'saved_models/5_border/history.csv'

plot_model_loss(pkl_path, csv_path)

