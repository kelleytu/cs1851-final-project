import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import torch

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


    def get_test_cancer_data(self):
        test_ids = np.load("../data/test1_ids.npy", allow_pickle=True)
        test_images = np.load("../data/test1_images.npy", allow_pickle=True)
        # train_labels = np.load("../data/test1_labels.npy", allow_pickle=True)
        test_tabular = np.load("../data/test1_tabular.npy", allow_pickle=True)
        metadata = np.load("../data/metadata.npy", allow_pickle=True)
        print("IDs Shape:", test_ids.shape)
        print("Images Shape:", test_images.shape)
        print("Tabular Shape:", test_tabular.shape)
        return test_ids, test_images, test_tabular, metadata

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
    print("images shape")
    print(images.shape)

    preprocess = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # now in opencv chn order

        # og = img.copy()
        # center crop, originally 224 x 224
        old_size = 224
        new_size = 204
        cut = (old_size - new_size) // 2
        
        cropped = img[cut:cut+new_size, cut:cut+new_size]

        # hair removal
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200) # mess around with this
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        eroded = cv2.erode(closed, kernel, iterations=1)

        lines = cv2.HoughLinesP(eroded, cv2.HOUGH_PROBABILISTIC, np.pi / 720, 35, 1, 5, 16) # from paper
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

                combined = np.hstack((cropped, edges, closed, eroded, line_img, line_mask, dilated, filled, img))
            else:            
                combined = np.hstack((cropped, edges, closed, eroded, img))

            cv2.imshow("preprocessing", combined)
            cv2.waitKey(0) 
            cv2.destroyAllWindows()

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

        # fill = gray.copy()
        # mask_coords = np.column_stack(np.where(dilated > 0))

        # for y, x in mask_coords:
        #     radius = 5
        #     img_length = 204
        #     y0 = max(0, y-radius)
        #     y1 = min(img_length, y + radius + 1)
        #     x0 = max(0, x-radius)
        #     x1 = min(img_length, x + radius + 1)

        #     patch = fill[y0:y1, x0:x1]
        #     patch_mask = dilated[y0:y1, x0:x1]
        #     neighbors = patch[patch_mask == 0]

        #     fill[y,x] = np.median(neighbors)
    return fill, line_mask, dilated