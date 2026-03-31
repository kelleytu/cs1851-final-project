import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

class FPDataLoader:
    def __init__(self):
        pass
                    
    def get_cancer_data(self):
        train_ids = np.load("../data/train_ids.npy", allow_pickle=True)
        train_images = np.load("../data/train_images.npy", allow_pickle=True)
        train_labels = np.load("../data/train_labels.npy", allow_pickle=True)
        train_tabular = np.load("../data/train_tabular.npy", allow_pickle=True)
        metadata = np.load("../data/metadata.npy", allow_pickle=True)
        print("IDs Shape:", train_ids.shape)
        print("Images Shape:", train_images.shape)
        print("Labels Shape:", train_labels.shape)
        print("Tabular Shape:", train_tabular.shape)
        return train_ids, train_images, train_labels, train_tabular, metadata

    def show_sample_imgs(self, train_images, train_labels):
        cancer_classes = np.unique(train_labels)
        for i, cancer_type in enumerate(cancer_classes):
            index = np.where(train_labels == cancer_type)[0][0]
            plt.subplot(2, 4, i + 1)
            plt.imshow(train_images[index])
            plt.title(cancer_type)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

def preprocess_images(images, power=6):
    # normalization
    # gray scale
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # now in opencv chn order
        
        cv2.imshow("original", img)

        # from paper 
        img = img.astype('float32')
        img_power = np.power(img, power)
        rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
        rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
        rgb_vec = rgb_vec/rgb_norm
        rgb_vec = 1/(rgb_vec*np.sqrt(3))
        img = np.multiply(img, rgb_vec)

        img = np.clip(img, 0, 255).astype(np.uint8)
        # img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        cv2.imshow("color const", img)

        cv2.waitKey(0)


   
    cv2.destroyAllWindows()
