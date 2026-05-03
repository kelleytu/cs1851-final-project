import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn as nn

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





