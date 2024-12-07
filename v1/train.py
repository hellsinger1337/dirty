import os
import shutil
import cv2
import numpy as np
import torch
import shutil
import heapq
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt


# -----------------------------
# Шаг 1: Предварительная Обработка Данных
# -----------------------------

def extract_aberration_features(image):
    """
    Извлекает дополнительные признаки из изображения.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Лапласиан
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.absolute(laplacian)
    laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())

    # Собель
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel = np.hypot(sobelx, sobely)
    sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min())

    # LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method='default')
    lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min())

    features = np.stack([laplacian, sobel, lbp], axis=-1)
    return features


def preprocess_mask(mask_path, target_size=(256, 256)):
    """
    Загружает маску и приводит её к бинарной форме.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask is None:
        raise ValueError(f"Не удалось загрузить маску по пути: {mask_path}")

    # Бинаризация: фон чёрный (0), все остальное белое (255)
    binary_mask = np.where(np.any(mask != [0, 0, 0], axis=-1), 255, 0).astype(np.uint8)

    # Изменение размера
    binary_mask = cv2.resize(binary_mask, target_size, interpolation=cv2.INTER_NEAREST)
    return binary_mask


def preprocess_and_save_images(input_dir, output_dir, target_size=(256, 256)):
    """
    Обрабатывает изображения и сохраняет их.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]

    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Извлечение дополнительных признаков
        features = extract_aberration_features(image)

        # Объединение изображения с признаками
        combined = np.concatenate([image, features], axis=-1)

        # Изменение размера
        combined_resized = cv2.resize(combined, target_size)

        # Сохранение
        save_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.npy')
        np.save(save_path, combined_resized)
        print(f"Сохранено обработанное изображение: {save_path}")


def preprocess_and_save_masks(input_dir, output_dir, target_size=(256, 256)):
    """
    Обрабатывает маски и сохраняет их.
    """
    os.makedirs(output_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    for mask_name in mask_files:
        mask_path = os.path.join(input_dir, mask_name)
        mask = preprocess_mask(mask_path, target_size)

        # Сохранение
        save_path = os.path.join(output_dir, os.path.splitext(mask_name)[0] + '.npy')
        np.save(save_path, mask)
        print(f"Сохранено обработанная маска: {save_path}")


# -----------------------------
# Шаг 2: Определение Класса Dataset
# -----------------------------

class PreprocessedContaminationDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

        if len(self.image_files) != len(self.mask_files):
            raise ValueError("Количество изображений и масок не совпадает.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        combined = np.load(image_path)
        mask = np.load(mask_path).astype(np.float32) / 255.0  # Нормализация маски

        # Преобразование в тензоры
        combined_tensor = torch.from_numpy(np.transpose(combined, (2, 0, 1))).float()  # [C, H, W]
        mask_tensor = torch.from_numpy(mask).float()  # [H, W]

        return combined_tensor, mask_tensor


# -----------------------------
# Шаг 3: Определение Архитектуры Модели
# -----------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64, 32)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        logits = self.outc(x)
        return logits


# -----------------------------
# Шаг 4: Основной Скрипт
# -----------------------------
def calculate_mIoU(val_loader, model, device, threshold=0.93):
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.int)

            # Предсказания модели
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > threshold).int()
            inter = 0.0
            union = 0.0
            for y_true, y_pred in zip(masks, preds):
                y_true_np = y_true.cpu().numpy()  
                y_pred_np = y_pred.cpu().numpy()  
                y_true_n = y_pred_np != 0
                y_pred_n = y_pred_np != 0
                # IoU для класса "грязь" (где значение 1)
                inter = (y_true_n & y_pred_n).sum()
                union = (y_true_n | y_pred_n).sum()                                          
                iou_score1 = inter / (union + 1e-8)
                y_true_n = y_pred_np == 0
                y_pred_n = y_pred_np == 0
                # IoU для класса "фон" (где значение 0)
                inter += (y_true_n & y_pred_n).sum()
                union += (y_true_n | y_pred_n).sum()
                iou_scores.append((iou_score1 +inter / (union + 1e-8))/2)
    return np.mean(iou_scores)

def recreate_datasets(PROCESSED_IMAGES_DIR, PROCESSED_MASKS_DIR, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, VAL_IMAGES_DIR, VAL_MASKS_DIR):
    for dir_path in [TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, VAL_IMAGES_DIR, VAL_MASKS_DIR]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)

    image_files = [f for f in os.listdir(PROCESSED_IMAGES_DIR) if os.path.isfile(os.path.join(PROCESSED_IMAGES_DIR, f))]
    mask_files = [f for f in os.listdir(PROCESSED_MASKS_DIR) if os.path.isfile(os.path.join(PROCESSED_MASKS_DIR, f))]

    assert len(image_files) == len(mask_files), "Количество изображений и масок не совпадает."

    train_images, val_images = train_test_split(image_files, train_size=0.8, random_state=42)

    for img_name in train_images:
        shutil.copy(os.path.join(PROCESSED_IMAGES_DIR, img_name), os.path.join(TRAIN_IMAGES_DIR, img_name))
        shutil.copy(os.path.join(PROCESSED_MASKS_DIR, img_name), os.path.join(TRAIN_MASKS_DIR, img_name))

    for img_name in val_images:
        shutil.copy(os.path.join(PROCESSED_IMAGES_DIR, img_name), os.path.join(VAL_IMAGES_DIR, img_name))
        shutil.copy(os.path.join(PROCESSED_MASKS_DIR, img_name), os.path.join(VAL_MASKS_DIR, img_name))

def main():   
    PROCESSED_IMAGES_DIR = './processed_data/images'
    PROCESSED_MASKS_DIR = './processed_data/masks'
    TRAIN_IMAGES_DIR = './IMAGES/train'
    TRAIN_MASKS_DIR = './MASKS/train'
    VAL_IMAGES_DIR = './IMAGES/val'
    VAL_MASKS_DIR = './MASKS/val'
    #preprocess_and_save_images("train_dataset/open_img",PROCESSED_IMAGES_DIR)
    #preprocess_and_save_masks("train_dataset/open_msk",PROCESSED_MASKS_DIR)
    MODELS_DIR = './models5'
    os.makedirs(MODELS_DIR, exist_ok=True)

    top_models = []

    model = UNet(n_channels=6, n_classes=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device('cpu')  
    model.load_state_dict(torch.load("models4/BaDmodel_7.pth", map_location=device))
    model.to(device)
    print(f"Модель загружена, начинаем дообучение")

    for epoch in range(1000):  
        if epoch % 10 == 0:
            recreate_datasets(PROCESSED_IMAGES_DIR, PROCESSED_MASKS_DIR, TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, VAL_IMAGES_DIR, VAL_MASKS_DIR)

            train_dataset = PreprocessedContaminationDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR)
            val_dataset = PreprocessedContaminationDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR)

            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(1), masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Эпоха {epoch + 1} завершена. Средний Loss: {avg_epoch_loss:.4f}")

        avg_val_mIoU = calculate_mIoU(val_loader, model, device,1-avg_epoch_loss)
        print(f"Эпоха {epoch + 1}: Средний валидационный mIoU: {avg_val_mIoU:.4f}")

        model_path = os.path.join(MODELS_DIR, f'BaDmodel_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_path)

        heapq.heappush(top_models, (avg_val_mIoU, model_path))
        if len(top_models) > 5:
            _, worst_model_path = heapq.heappop(top_models)
            os.remove(worst_model_path) 
            print(f"Удалена модель с худшим mIoU: {worst_model_path}")

    print("Обучение завершено. Топ-5 моделей сохранены в директории 'models'.")


if __name__ == "__main__":
    main()