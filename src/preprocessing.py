import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms as T
import os
import joblib

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


    def get_test_cancer_data(self, test_set):
        test_ids = np.load(f"../data/test{test_set}_ids.npy", allow_pickle=True)
        test_images = np.load(f"../data/test{test_set}_images.npy", allow_pickle=True)
        test_tabular = np.load(f"../data/test{test_set}_tabular.npy", allow_pickle=True)
        print("IDs Shape:", test_ids.shape)
        print("Images Shape:", test_images.shape)
        print("Tabular Shape:", test_tabular.shape)
        return test_ids, test_images, test_tabular

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



def preprocess_images(images, power=6, show=False):
    # normalization
    # gray scale
    save_dir = os.path.join("..", "preprocess")
    os.makedirs(save_dir, exist_ok=True)

    print("images shape")
    print(images.shape)

    preprocess = []

    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # now in opencv chn order

        og = img.copy()
        # center crop, originally 224 x 224
        # # old_size = 224
        # # new_size = 204
        # # cut = (old_size - new_size) // 2
        
        # # cropped = img[cut:cut+new_size, cut:cut+new_size]
        
        cropped = crop_img(img)


        # hair removal
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200) # mess around with this
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        eroded = cv2.erode(closed, kernel, iterations=1)

        lines = cv2.HoughLinesP(closed, cv2.HOUGH_PROBABILISTIC, np.pi / 720, 35, 1, 5, 16) # from paper
        # print(f"num lines found {lines}")
        if lines is not None:
            filled, line_mask, dilated = remove_fill_lines(cropped, gray, lines, 1, 300, 3)
            img = filled
        else:
            img = cropped


        # color constancy from paper 
        img = img.astype('float32')
        img_power = np.power(img, power)
        rgb_vec = np.power(np.mean(img_power, (0,1)), 1/power)
        rgb_norm = np.sqrt(np.sum(np.power(rgb_vec, 2.0)))
        rgb_vec = rgb_vec/rgb_norm
        rgb_vec = 1/(rgb_vec*np.sqrt(3))
        img = np.multiply(img, rgb_vec)

        img = np.clip(img, 0, 255).astype(np.uint8)
        # img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        if show:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            closed = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
            eroded = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)

            if lines is not None:
                line_img = draw_lines(lines,cropped.copy())
                line_mask = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR)
                dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
                # filled = cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR)

                combined = np.hstack((og, cropped, edges, closed, eroded, line_img, line_mask, dilated, filled, img))
            else:            
                combined = np.hstack((og, cropped, edges, closed, eroded, img))

            while True:
                cv2.imshow("preprocessing", combined)
                key = cv2.waitKey(0) & 0xFF

                if key == 27:
                    cv2.destroyAllWindows()
                    exit()
                elif key == ord("s"):
                    filename = os.path.join(save_dir, f"image_{i}.png")
                    cv2.imwrite(filename, combined)
                    print(f"Saved {filename}")
                
                cv2.destroyAllWindows()
                break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        preprocess.append(img)
    
    preprocess = np.array(preprocess)
    print("preprocess shape")
    print(preprocess.shape)

    return preprocess

def change_img_format(images):
    images = np.transpose(images, (0, 3, 1, 2))
    images = torch.tensor(images).float()

    return images


def draw_lines(lines, img):
    for i in range(0, len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
    return img


def remove_fill_lines(img, gray, lines, min_len, max_len, mask_thickness):
    line_mask = np.zeros_like(gray)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length <= max_len and length >= min_len:
            cv2.line(line_mask, (x1, y1),(x2, y2), 255, thickness=mask_thickness)
        
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dilated = cv2.dilate(line_mask, dilate_kernel) # can change iterations

    fill = cv2.inpaint(img, dilated, 3, cv2.INPAINT_TELEA)
    return fill, line_mask, dilated



def transform_data(X, train=True, show=False):
    
    if train:
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                # T.RandomRotation(90),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
    else:
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )


    augmented = []
    for img in X:
        augmented.append(transform(img))

    if show:
        original_np = X[0]
        augmented_np = augmented[0].permute(1,2,0).cpu().numpy()

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
        
    return torch.stack(augmented)

def crop_img(img, crop_size=204, ring_width=20):
        h, w = img.shape[:2]

        original_h, original_w = h, w

        y0 = (h - crop_size) // 2
        x0 = (w - crop_size) // 2
        y1 = y0 + crop_size
        x1 = x0 + crop_size

        cropped = img[y0:y1, x0:x1]

        ring_mask = np.zeros((crop_size, crop_size), dtype=bool)
        ring_mask[:ring_width, :] = True
        ring_mask[-ring_width:, :] = True
        ring_mask[:, :ring_width] = True
        ring_mask[:, -ring_width:] = True

        ring_pixels = cropped[ring_mask]
        median_color = np.median(ring_pixels, axis=0).astype(np.uint8)

        top = y0
        bottom = original_h - crop_size - top
        left = x0
        right = original_w - crop_size - left

        padded = cv2.copyMakeBorder(
            cropped,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=median_color.tolist()
        )

        return padded

def build_tab_preprocessor():
    tabular_preprocessor = ColumnTransformer(
        transformers=[
            ("age", StandardScaler(), [0]),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), [1,2])
        ]
    )
    return tabular_preprocessor

def preprocess_tabular(X_tab_train, X_tab_test, save_path="tabular_preprocessor.pkl"):
    tabular_preprocessor = build_tab_preprocessor()
    X_tab_train = tabular_preprocessor.fit_transform(X_tab_train)
    X_tab_test = tabular_preprocessor.transform(X_tab_test)

    if hasattr(X_tab_test, "toarray"):
        X_tab_test = X_tab_test.toarray()
    if hasattr(X_tab_train, "toarray"):
        X_tab_train = X_tab_train.toarray()
    joblib.dump(tabular_preprocessor, save_path)

    
    return X_tab_train, X_tab_test, tabular_preprocessor

def preprocess_tabular_test(X_tab_test, save_path="tabular_preprocessor.pkl"):
    tabular_preprocessor = joblib.load(save_path)
    X_tab_test = tabular_preprocessor.transform(X_tab_test)

    if hasattr(X_tab_test, "toarray"):
        X_tab_test = X_tab_test.toarray()
    return X_tab_test

    

