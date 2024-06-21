import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    RandomBrightnessContrast,
    ElasticTransform,
    RandomCrop,
)

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    train_x = sorted(list(map(lambda x: x.replace('\\', '/'), glob(os.path.join(path, "training", "images", "*")))))
    train_y = sorted(list(map(lambda x: x.replace('\\', '/'), glob(os.path.join(path, "training", "mask", "*")))))

    test_x = sorted(list(map(lambda x: x.replace('\\', '/'), glob(os.path.join(path, "test", "images", "*")))))
    test_y = sorted(list(map(lambda x: x.replace('\\', '/'), glob(os.path.join(path, "test", "mask", "*")))))

    return (train_x, train_y), (test_x, test_y)

    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        name = x.split("/")[-1].split(".")[0]

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        # if augment == True:
        #     aug = HorizontalFlip(p=1.0)
        #     augmented = aug(image=x, mask=y)
        #     x1 = augmented["image"]
        #     y1 = augmented["mask"]

        #     aug = VerticalFlip(p=1.0)
        #     augmented = aug(image=x, mask=y)
        #     x2 = augmented["image"]
        #     y2 = augmented["mask"]

        #     aug = Rotate(limit=45, p=1.0)
        #     augmented = aug(image=x, mask=y)
        #     x3 = augmented["image"]
        #     y3 = augmented["mask"]

        #     aug = RandomBrightnessContrast(p=1.0)
        #     augmented = aug(image=x, mask=y)
        #     x4 = augmented["image"]
        #     y4 = augmented["mask"]

        #     aug = ElasticTransform(p=1.0, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
        #     augmented = aug(image=x, mask=y)
        #     x5 = augmented["image"]
        #     y5 = augmented["mask"]
        #     aug = Rotate(limit=90, p=1.0)  # Rotación adicional de 90 grados
        #     augmented = aug(image=x, mask=y)
        #     x7 = augmented["image"]
        #     y7 = augmented["mask"]
            # aug = RandomCrop(p=1.0, height=256, width=256)
            # augmented = aug(image=x, mask=y)
            # x6 = augmented["image"]
            # y6 = augmented["mask"]

        X = [x]
        Y = [y]

        # else:
        #     X = [x]
        #     Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
            #print(f"Guardado: {image_path} - {mask_path}")
            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = r"D:\ruebastesis\ruebaidiot\DRIVE17"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Create directories to save the augmented data """
    create_dir("new_data17_3/train/image/")
    create_dir("new_data17_3/train/mask/")
    create_dir("new_data17_3/test/image/")
    create_dir("new_data17_3/test/mask/")

    """ Data augmentation """
    augment_data(train_x, train_y, r"D:\ruebastesis\ruebaidiot\new_data17_3\train", augment=True)
    augment_data(test_x, test_y, r"D:\ruebastesis\ruebaidiot\new_data17_3\test", augment=True)
