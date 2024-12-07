import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern

# -----------------------------
# Шаг 1: Предварительная Обработка Данных
# -----------------------------

def extract_aberration_features(image):
    """
    Извлекает дополнительные признаки из изображения.
    
    Args:
        image (np.ndarray): Изображение в формате RGB.
    
    Returns:
        np.ndarray: Дополнительные признаки с формой [H, W, 3].
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
    
    features = np.stack([laplacian, sobel, lbp], axis=-1)  # [H, W, 3]
    return features

def preprocess_mask(mask_path):
    """
    Загружает маску, преобразует все ненулевые пиксели в белые, а черные оставляет без изменений.
    
    Args:
        mask_path (str): Путь к файлу маски.
    
    Returns:
        np.ndarray: Обработанная бинарная маска.
    """
    # Загрузка маски в цвете
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask is None:
        raise ValueError(f"Не удалось загрузить маску по пути: {mask_path}")
    
    # Проверка, что маска имеет 3 канала
    if len(mask.shape) != 3 or mask.shape[2] != 3:
        raise ValueError(f"Маска должна быть цветной (3 канала). Получено: {mask.shape}")
    
    # Создание бинарной маски: черный фон (0), все остальное - белый (255)
    binary_mask = np.where(np.any(mask != [0, 0, 0], axis=-1), 255, 0).astype(np.uint8)
    
    return binary_mask

def preprocess_and_save_images(input_dir, output_dir, transform=None):
    """
    Предварительно обрабатывает все изображения и сохраняет их в формате .npy.
    
    Args:
        input_dir (str): Путь к исходной директории с изображениями.
        output_dir (str): Путь к директории для сохранения обработанных изображений.
        transform (albumentations.Compose, optional): Дополнительные трансформации.
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
        features = extract_aberration_features(image)  # [H, W, 3]
        
        # Объединение оригинального изображения с признаками
        combined = np.concatenate([image, features], axis=-1)  # [H, W, 6]
        
        # Применение трансформаций, если они заданы
        if transform:
            augmented = transform(image=combined)
            combined = augmented['image']  # [H, W, 6]
        
        # Сохранение как .npy
        save_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + '.npy')
        np.save(save_path, combined)
        print(f"Сохранено обработанное изображение: {save_path}")

def preprocess_and_save_masks(input_dir, output_dir, transform=None):
    """
    Предварительно обрабатывает все маски и сохраняет их в формате .npy.
    
    Args:
        input_dir (str): Путь к исходной директории с масками.
        output_dir (str): Путь к директории для сохранения обработанных масок.
        transform (albumentations.Compose, optional): Дополнительные трансформации.
    """
    os.makedirs(output_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    for mask_name in mask_files:
        mask_path = os.path.join(input_dir, mask_name)
        mask = preprocess_mask(mask_path)
        
        # Применение трансформаций, если они заданы
        if transform:
            augmented = transform(image=mask)
            mask = augmented['image'].squeeze().astype(np.uint8)
        
        # Сохранение как .npy
        save_path = os.path.join(output_dir, os.path.splitext(mask_name)[0] + '.npy')
        np.save(save_path, mask)
        print(f"Сохранено обработанное маска: {save_path}")

def preprocess_data():
    """
    Выполняет предварительную обработку всех изображений и масок.
    """
    # Пути к исходным данным
    IMAGES_DIR = './train_dataset/cv_synt_dataset/synt_img'  # Замените на ваш путь к изображениям
    MASKS_DIR = './train_dataset/cv_synt_dataset/synt_msk'   # Замените на ваш путь к маскам
    
    # Пути к обработанным данным
    PROCESSED_IMAGES_DIR = './processed_data/images'
    PROCESSED_MASKS_DIR = './processed_data/masks'
    
    # Определение трансформаций для предварительной обработки изображений
    preprocess_image_transform = A.Compose([
        A.Resize(256, 256),
        # Удаляем A.Normalize, чтобы избежать двойной нормализации
    ])
    
    # Определение трансформаций для предварительной обработки масок
    preprocess_mask_transform = A.Compose([
        A.Resize(256, 256)
        # Дополнительные трансформации для масок можно добавить здесь, если необходимо
    ])
    
    print("Начало предварительной обработки изображений...")
    preprocess_and_save_images(IMAGES_DIR, PROCESSED_IMAGES_DIR, transform=preprocess_image_transform)
    print("Предварительная обработка изображений завершена.\n")
    
    print("Начало предварительной обработки масок...")
    preprocess_and_save_masks(MASKS_DIR, PROCESSED_MASKS_DIR, transform=preprocess_mask_transform)
    print("Предварительная обработка масок завершена.\n")

# -----------------------------
# Шаг 2: Определение Класса Dataset
# -----------------------------

class PreprocessedContaminationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Инициализирует датасет для предварительно обработанных данных.
        
        Args:
            image_dir (str): Директория с предварительно обработанными изображениями (.npy).
            mask_dir (str): Директория с предварительно обработанными масками (.npy).
            transform (albumentations.Compose, optional): Трансформации для данных.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.npy')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])
        
        # Проверка соответствия количества изображений и масок
        if len(self.image_files) != len(self.mask_files):
            print(f"Количество изображений: {len(self.image_files)}")
            print(f"Количество масок: {len(self.mask_files)}")
            print("Количество изображений и масок не совпадает.")
            # Перечислим отсутствующие маски
            image_basenames = set(os.path.splitext(f)[0] for f in self.image_files)
            mask_basenames = set(os.path.splitext(f)[0] for f in self.mask_files)
            missing_masks = image_basenames - mask_basenames
            if missing_masks:
                print("Отсутствующие маски для следующих изображений:")
                for name in missing_masks:
                    print(f"{name}.npy")
            raise ValueError("Количество изображений и масок не совпадает.")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Загрузка обработанного изображения
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        combined = np.load(image_path)  # [256, 256, 6]
        
        # Загрузка обработанной маски
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = np.load(mask_path)  # [256, 256]
        
        # Преобразование маски в формат float32
        mask = mask.astype(np.float32) / 255.0  # [256, 256]
        
        if self.transform:
            augmented = self.transform(image=combined, mask=mask)
            combined = augmented['image']  # [C, H, W] после ToTensorV2
            mask = augmented['mask']      # [H, W]
        
        return combined, mask

# -----------------------------
# Шаг 3: Определение Архитектуры U-Net
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
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(32, 64)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
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
        x1 = self.inc(x)        # [B, 32, H, W]
        x2 = self.down1(x1)     # [B, 64, H/2, W/2]
        x3 = self.down2(x2)     # [B, 128, H/4, W/4]
        x4 = self.down3(x3)     # [B, 256, H/8, W/8]
        x5 = self.down4(x4)     # [B, 512, H/16, W/16]
        
        x = self.up1(x5)        # [B, 256, H/8, W/8]
        x = torch.cat([x, x4], dim=1)  # [B, 512, H/8, W/8]
        x = self.conv1(x)       # [B, 256, H/8, W/8]
        
        x = self.up2(x)         # [B, 128, H/4, W/4]
        x = torch.cat([x, x3], dim=1)  # [B, 256, H/4, W/4]
        x = self.conv2(x)       # [B, 128, H/4, W/4]
        
        x = self.up3(x)         # [B, 64, H/2, W/2]
        x = torch.cat([x, x2], dim=1)  # [B, 128, H/2, W/2]
        x = self.conv3(x)       # [B, 64, H/2, W/2]
        
        x = self.up4(x)         # [B, 32, H, W]
        x = torch.cat([x, x1], dim=1)  # [B, 64, H, W]
        x = self.conv4(x)       # [B, 32, H, W]
        
        logits = self.outc(x)   # [B, n_classes, H, W]
        return logits

# -----------------------------
# Шаг 4: Функции для Вычисления Метрик и Визуализации
# -----------------------------

def calculate_mIoU(loader, model, threshold=0.5):
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        iou_scores = []
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.int)
                
                outputs = model(images)
                outputs = torch.sigmoid(outputs)
                preds = (outputs > threshold).int()
                
                intersection = (preds & masks).float().sum((1, 2))
                union = (preds | masks).float().sum((1, 2))
                iou = (intersection + 1e-6) / (union + 1e-6)
                
                iou_scores.extend(iou.cpu().numpy())
        
        mIoU = np.mean(iou_scores)
        return mIoU

def visualize_predictions(loader, model, num_images=5):
    """
    Визуализирует предсказания модели на примере изображений из загрузчика данных.
    
    Args:
        loader (DataLoader): Загрузчик данных.
        model (nn.Module): Модель для инференса.
        num_images (int): Количество изображений для визуализации.
    """
    model.eval()
    images_shown = 0
    device = torch.device('cpu')  # Используем CPU
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            outputs = model(images)
            outputs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()  # [B, H, W]
            masks = masks.cpu().numpy()
            images = images.cpu().numpy()
            
            for i in range(images.shape[0]):
                if images_shown >= num_images:
                    return
                img = images[i][:3]  # Первые 3 канала RGB
                img = np.transpose(img, (1, 2, 0))  # [H, W, C]
                mask = masks[i]
                pred = outputs[i]
                
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(img)
                plt.title('Исходное изображение')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(mask, cmap='gray')
                plt.title('Истинная маска')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(pred, cmap='gray')
                plt.title('Предсказанная маска')
                plt.axis('off')
                
                plt.show()
                images_shown += 1

def format_time(seconds):
    """Форматирует время в формате ЧЧ:ММ:СС."""
    h = math.floor(seconds / 3600)
    m = math.floor((seconds % 3600) / 60)
    s = seconds % 60
    return f"{h}h {m}m {s:.2f}s"

# -----------------------------
# Шаг 5: Основной Скрипт Обучения
# -----------------------------

def main():
    # Шаг 1: Предварительная Обработка Данных
    preprocess_data()
    
    # Пути к обработанным данным
    PROCESSED_IMAGES_DIR = './processed_data/images'
    PROCESSED_MASKS_DIR = './processed_data/masks'
    
    # Создание директорий для обучающей и валидационной выборок
    TRAIN_SIZE = 0.6  # Доля данных для обучения
    image_files = sorted([f for f in os.listdir(PROCESSED_IMAGES_DIR) if f.endswith('.npy')])
    mask_files = sorted([f for f in os.listdir(PROCESSED_MASKS_DIR) if f.endswith('.npy')])
    
    # Проверка соответствия количества изображений и масок
    if len(image_files) != len(mask_files):
        print(f"Количество изображений: {len(image_files)}")
        print(f"Количество масок: {len(mask_files)}")
        print("Количество изображений и масок не совпадает.")
        # Перечислим отсутствующие маски
        image_basenames = set(os.path.splitext(f)[0] for f in image_files)
        mask_basenames = set(os.path.splitext(f)[0] for f in mask_files)
        missing_masks = image_basenames - mask_basenames
        if missing_masks:
            print("Отсутствующие маски для следующих изображений:")
            for name in missing_masks:
                print(f"{name}.npy")
        raise ValueError("Количество изображений и масок не совпадает.")
    
    # Разделение на обучающую и валидационную выборки
    train_images, val_images = train_test_split(image_files, train_size=TRAIN_SIZE, random_state=42)
    
    # Создание директорий для обучающей и валидационной выборок
    TRAIN_IMAGES_DIR = os.path.join(PROCESSED_IMAGES_DIR, 'train')
    TRAIN_MASKS_DIR = os.path.join(PROCESSED_MASKS_DIR, 'train')
    VAL_IMAGES_DIR = os.path.join(PROCESSED_IMAGES_DIR, 'val')
    VAL_MASKS_DIR = os.path.join(PROCESSED_MASKS_DIR, 'val')
    
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAIN_MASKS_DIR, exist_ok=True)
    os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
    os.makedirs(VAL_MASKS_DIR, exist_ok=True)
    
    # Копирование файлов в соответствующие директории
    for img_name in train_images:
        src_img_path = os.path.join(PROCESSED_IMAGES_DIR, img_name)
        dst_img_path = os.path.join(TRAIN_IMAGES_DIR, img_name)
        shutil.copy(src_img_path, dst_img_path)
        
        mask_name = img_name  # Предполагается, что имена масок совпадают с именами изображений
        src_mask_path = os.path.join(PROCESSED_MASKS_DIR, mask_name)
        dst_mask_path = os.path.join(TRAIN_MASKS_DIR, mask_name)
        if os.path.exists(src_mask_path):
            shutil.copy(src_mask_path, dst_mask_path)
        else:
            print(f"Маска не найдена для изображения: {img_name}")
    
    for img_name in val_images:
        src_img_path = os.path.join(PROCESSED_IMAGES_DIR, img_name)
        dst_img_path = os.path.join(VAL_IMAGES_DIR, img_name)
        shutil.copy(src_img_path, dst_img_path)
        
        mask_name = img_name  # Предполагается, что имена масок совпадают с именами изображений
        src_mask_path = os.path.join(PROCESSED_MASKS_DIR, mask_name)
        dst_mask_path = os.path.join(VAL_MASKS_DIR, mask_name)
        if os.path.exists(src_mask_path):
            shutil.copy(src_mask_path, dst_mask_path)
        else:
            print(f"Маска не найдена для изображения: {img_name}")
    
    # Создание файла data.yaml (если требуется, например, для YOLO или других фреймворков)
    data_yaml_content = f"""
train: {os.path.abspath(TRAIN_IMAGES_DIR)}
val: {os.path.abspath(VAL_IMAGES_DIR)}

nc: 1  # Количество классов (1 для загрязнения)
names: ['contaminated']  # Название класса
"""
    with open('data.yaml', 'w') as f:
        f.write(data_yaml_content)
    
    print("Файл data.yaml создан.\n")
    print("Данные разделены на обучающую и валидационную выборки.\n")
    
    # Шаг 2: Создание Датасетов и Загрузчиков
    train_dataset = PreprocessedContaminationDataset(
        image_dir=TRAIN_IMAGES_DIR,
        mask_dir=TRAIN_MASKS_DIR,
        transform=None  # Трансформации будут применяться через отдельные трансформы
    )
    
    val_dataset = PreprocessedContaminationDataset(
        image_dir=VAL_IMAGES_DIR,
        mask_dir=VAL_MASKS_DIR,
        transform=None
    )
    
    # Определение трансформаций (аугментаций) для тренировочного датасета
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(
            mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 6 каналов
            std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)    # 6 каналов
        ),
        ToTensorV2()
    ])
    
    # Трансформации для валидационного датасета (без аугментаций)
    val_transform = A.Compose([
        A.Normalize(
            mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 6 каналов
            std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)    # 6 каналов
        ),
        ToTensorV2()
    ])
    
    # Применение трансформаций
    train_dataset.transform = train_transform
    val_dataset.transform = val_transform
    
    # Создание загрузчиков данных
    # На Windows рекомендуется устанавливать num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    
    # Шаг 3: Инициализация Модели, Критерия и Оптимизатора
    model = UNet(n_channels=6, n_classes=1) 
    device = torch.device('cpu')  
    model.load_state_dict(torch.load("model80.pth", map_location=device))
     # Используем CPU
    print(f'Используется устройство: {device}\n')
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Шаг 4: Цикл Обучения
    num_epochs = 80
    best_mIoU = 0.0
    epoch_times = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Время начала эпохи
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            outputs = outputs.squeeze(1)  # [B, H, W]
            
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        # Валидация
        val_mIoU = calculate_mIoU(val_loader, model)
        print(f'Validation mIoU: {val_mIoU:.4f}')
        
        # Сохранение лучшей модели
        if val_mIoU > best_mIoU:
            best_mIoU = val_mIoU
            torch.save(model.state_dict(), 'model.pth')
            print(f'Лучшая модель сохранена с mIoU: {best_mIoU:.4f}')
        
        # (Опционально) Визуализация предсказаний
        # visualize_predictions(val_loader, model, num_images=1)
        
        # Подсчет времени эпохи
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        
        # Вычисление среднего времени за прошедшие эпохи
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        
        # Оценка оставшегося времени
        remaining_epochs = num_epochs - (epoch + 1)
        remaining_time = avg_epoch_time * remaining_epochs
        formatted_remaining_time = format_time(remaining_time)
        
        # Вывод информации о времени
        print(f"Время эпохи: {format_time(epoch_duration)}")
        print(f"Оставшееся время обучения: {formatted_remaining_time}\n")
    
    print(f'Обучение завершено. Лучшая модель имеет mIoU: {best_mIoU:.4f}')
    
    # Окончательное сохранение модели (если не была сохранена в цикле)
    if not os.path.exists('model.pth'):
        torch.save(model.state_dict(), 'model.pth')
        print("Модель успешно сохранена как model.pth")
    
    # Шаг 5: Инференс на Отдельных Изображениях (Опционально)
    def infer_image(image_path, model, device, output_path='./mask_image.png', threshold=0.5):
        """
        Выполняет инференс на одном изображении и сохраняет маску.
        
        Args:
            image_path (str): Путь к входному изображению.
            model (nn.Module): Обученная модель.
            device (torch.device): Устройство для инференса.
            output_path (str): Путь для сохранения маски.
            threshold (float): Порог для бинаризации предсказаний.
        """
        model.eval()
        with torch.no_grad():
            image = cv2.imread(image_path)
            if image is None:
                print(f"Не удалось загрузить изображение: {image_path}")
                return
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Извлечение дополнительных признаков
            features = extract_aberration_features(image_rgb)  # [H, W, 3]
            
            # Объединение исходного изображения с признаками
            combined = np.concatenate([image_rgb, features], axis=-1)  # [H, W, 6]
            
            # Применение трансформаций
            transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(
                    mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                ToTensorV2()
            ])
            
            augmented = transform(image=combined)
            image_tensor = augmented['image'].unsqueeze(0).to(device, dtype=torch.float32)  # [1, 6, 256, 256]
            
            # Инференс
            output = model(image_tensor)
            output = torch.sigmoid(output).squeeze(1).cpu().numpy()  # [256, 256]
            
            # Пороговая обработка
            mask = (output > threshold).astype(np.uint8) * 255  # [256, 256]
            
            # Масштабирование маски до оригинального размера
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Сохранение маски
            cv2.imwrite(output_path, mask_resized)
            print(f"Маска сохранена по пути: {output_path}")
    
    # Пример использования функции инференса (раскомментируйте и замените путь к изображению)
    # infer_image('./path_to_test_image/test_image.png', model, device)

if __name__ == '__main__':
    main()