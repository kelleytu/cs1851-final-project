import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn as nn

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import sparse
import numpy as np
import seaborn as sns

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class FocalLoss(nn.Module):
    def __init__(self, weights, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weights = weights
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, y):
        criterion = nn.CrossEntropyLoss(weight=self.weights)
        ce_loss = criterion(logits,y)

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()

class FusionDataset(Dataset):
    def __init__(self, X_image, X_tabular, y):
        self.X_image = X_image
        self.X_tabular = X_tabular
        self.y = y

    def __len__(self):
        return len(self.X_image)

    def __getitem__(self, index):
        return self.X_image[index], self.X_tabular[index], self.y[index]
    
def visualize_augmentation(image_tensor):
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor()
        ]
    )

    augmented = transform(image_tensor)
    original_np = image_tensor.permute(1,2,0).cpu().numpy()
    augmented_np = augmented.permute(1,2,0).cpu().numpy()

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(original_np)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(augmented_np)
    plt.title("Augmented")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_embedding(data, labels, title, random_state=0, save=False, filename=None):
    """
    data: (N, D) numpy array
    labels: (N,) array-like (ints)
    """
    if sparse.issparse(data):
        data = data.toarray()

    data = np.asarray(data)
    labels = np.asarray(labels).reshape(-1)


    data_50 = PCA(n_components=min(50, data.shape[1]), random_state=random_state).fit_transform(data)
    reduced = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(data_50)

    plt.figure(figsize=(6, 4))
    
    classes = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10", len(classes))
    
    for i, c in enumerate(classes):
        # get all of one class
        idx = labels == c
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f"class {c}", color=cmap(i), s=5)


    # plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=3)
    plt.legend()
    plt.title(f"tsne projection of {title}")
    plt.tight_layout()

    if save:
        plt.savefig(filename, dpi=200)
    plt.show()

def grad_cam(model, X_img, X_tab, original_X_img, y_true):
    model.eval()
    wrapped_model = WrappedModel(model, X_tab)
    target_layer = [model.image_classifier.layer4[-1]]

    with torch.no_grad():
        logits = model(X_img, X_tab)
        y_pred = logits.argmax(dim=1).item()
    
    targets = [ClassifierOutputTarget(y_pred)]

    with GradCAM(model=wrapped_model, target_layers=target_layer) as cam:
        grayscale_cam = cam(input_tensor=X_img, targets=targets)[0]
        img_np = X_img.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean

        img_np = np.clip(img_np, 0, 1).astype(np.float32)

        grayscale_cam = np.squeeze(grayscale_cam).astype(np.float32)

        visualization = show_cam_on_image(
            img_np,
            grayscale_cam,
            use_rgb=True
        )
        
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(original_X_img)
        plt.title("Original image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title(f"Grad-CAM | Predicted = {y_pred}, Actual = {y_true}")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
    
class WrappedModel(torch.nn.Module):
    def __init__(self, model, X_tab):
        super().__init__()
        self.model = model
        self.X_tab = X_tab
    
    def forward(self, X_img):
        return self.model(X_img, self.X_tab)
        

def plot_cm(cm):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix of Best Model")
    plt.show()
