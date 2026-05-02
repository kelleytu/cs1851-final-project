import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as T

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






# def visualize_activations(model, input_tensor, layer_name="conv1"):
#     model.eval()
#     activations = {}
    
#     def hook_fn(module, input, output):
#         activations[layer_name] = output.detach().cpu()

#     target_layer = None
#     for name, module in model.named_modules():
#         if name == layer_name:
#             target_layer = module
#             break
    
#     if target_layer is None:
#         raise ValueError(f"Layer {layer_name} not found in the model.")
    
#     hook = target_layer.register_forward_hook(hook_fn)
#     with torch.no_grad():
#         if input_tensor.dim() == 3:
#             input_tensor = input_tensor.unsqueeze(0)
#         if input_tensor.dim() == 2:
#             input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
#         device = next(model.parameters()).device
#         input_tensor = input_tensor.to(device)
#         _ = model(input_tensor)
#     hook.remove()

#     feature_maps = activations[layer_name][0]
#     num_maps = min(feature_maps.shape[0], 16)
#     cols = 4
#     rows = -(-num_maps // cols)
#     plt.figure(figsize=(12, 3 * rows))
#     for i in range(num_maps):
#         plt.subplot(rows, cols, i + 1)
#         plt.imshow(feature_maps[i].cpu(), cmap='viridis')
#         plt.axis('off')
#         plt.title(f"{layer_name} - Map {i+1}")
#     plt.tight_layout()
#     plt.show() 