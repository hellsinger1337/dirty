import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.feature import local_binary_pattern

# Определение модели (должно совпадать с архитектурой при обучении)
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

# Функция для извлечения дополнительных признаков
def extract_aberration_features(image):
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

# Функция для загрузки модели
def load_model(model_path, device):
    model = UNet(n_channels=6, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Функция для предобработки изображения
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Извлечение дополнительных признаков
    features = extract_aberration_features(image_rgb)
    
    # Комбинирование исходного изображения с признаками
    combined = np.concatenate([image_rgb, features], axis=-1)  # [H, W, 6]
    
    # Применение трансформаций
    transform = A.Compose([
        A.Resize(256, 256),
         A.Normalize(
            mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 6 элементов для 6 каналов
            std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)     # 6 элементов для 6 каналов
        ),
        ToTensorV2()
    ])
    
    augmented = transform(image=combined)
    image_tensor = augmented['image'].unsqueeze(0)  # [1, 6, 256, 256]
    
    return image_tensor, image.shape[:2]  # Возвращаем оригинальный размер

# Функция для постобработки маски
def postprocess_mask(mask_tensor, original_size, threshold=0.5):
    mask = torch.sigmoid(mask_tensor)
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > threshold).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return mask_resized

# Основная функция инференса
def main():
    
    model_path = "model30(29).pth"
    image_path = "test.jpg"
    
    device = torch.device('cpu')  # Используем CPU
    
    model = load_model(model_path, device)
    
    # Предобработка изображения
    input_tensor, original_size = preprocess_image(image_path)
    input_tensor = input_tensor.to(device, dtype=torch.float32)
    
    # Инференс
    with torch.no_grad():
        output = model(input_tensor)
    
    # Постобработка маски
    mask = postprocess_mask(output, original_size)
    
    # Сохранение маски
    output_mask_path = os.path.splitext(image_path)[0] + '_mask.png'
    cv2.imwrite(output_mask_path, mask)
    print(f"Маска сохранена по пути: {output_mask_path}")

if __name__ == "__main__":
    main()