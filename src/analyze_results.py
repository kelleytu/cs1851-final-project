import pickle
import matplotlib.pyplot as plt

# Open the file in 'rb' (read binary) mode
with open('resnet_fusion_history.pkl', 'rb') as file:
    # Load the object from the file
    history = pickle.load(file)

# print(history)
# train loss, test loss test acc, f1
metrics_list = ["test_loss", "test_acc", "test_f1"]

fig, axes = plt.subplots(1,3, figsize=(15,4))
num_epochs = len(history["test_loss"])

for i, m in enumerate(metrics_list):
    axes[i].plot(history[m])
    axes[i].set_title(f"{m} over {num_epochs} epochs")
    axes[i].set_xlabel("epoch")
    axes[i].set_ylabel(m)

plt.tight_layout()
plt.show()
