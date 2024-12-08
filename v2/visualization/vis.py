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
import matplotlib.pyplot as plt

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


def extract_aberration_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   
    
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.absolute(laplacian)
    laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())    
    
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel = np.hypot(sobelx, sobely)
    sobel = (sobel - sobel.min()) / (sobel.max() - sobel.min())    
    
    lbp = local_binary_pattern(gray, P=8, R=1, method='default')
    lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min())
    
    features = np.stack([laplacian, sobel, lbp], axis=-1)  
    return features


def load_model(model_path, device):
    model = UNet(n_channels=6, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Изображение по пути {image_path} не может быть загружено.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
    features = extract_aberration_features(image_rgb)
       
    combined = np.concatenate([image_rgb, features], axis=-1)  
     
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(
            mean=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  
            std=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)     
        ),
        ToTensorV2()
    ])
    
    augmented = transform(image=combined)
    image_tensor = augmented['image'].unsqueeze(0)  
    
    return image_tensor, image.shape[:2], image_rgb  


def postprocess_mask(mask_tensor, original_size, gamma=2.0, focus_min=0.50, focus_max=1.0):
    mask = torch.sigmoid(mask_tensor)
    mask = mask.squeeze().cpu().numpy()
    
    
    mask = np.clip(mask, focus_min, focus_max)
    mask = (mask - focus_min) / (focus_max - focus_min)
    mask = np.power(mask, gamma)
    mask = (mask * 255).astype(np.uint8)
    
    
    mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    return mask_resized


def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.5):
    colored_mask = np.zeros_like(image)
    colored_mask[:, :] = color  

    
    mask_binary = mask > 127

    
    overlay = image.copy()
    overlay[mask_binary] = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)[mask_binary]

    return overlay


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Обработка одного изображения и наложение маски 'грязи'.")
    parser.add_argument('image_path', type=str, help='Путь к входному изображению.')
    parser.add_argument('--model_path', type=str, default='../model80.pth', help='Путь к модели.')
    parser.add_argument('--color', type=str, default='255,0,0', help='Цвет маски в формате R,G,B (по умолчанию: красный).')
    parser.add_argument('--alpha', type=float, default=0.5, help='Прозрачность маски (0.0 - 1.0).')
    args = parser.parse_args()

    image_path = args.image_path
    model_path = args.model_path
    color = tuple(map(int, args.color.split(',')))
    alpha = args.alpha

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    if not os.path.exists(image_path):
        print(f"Изображение по пути {image_path} не существует.")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Модель по пути {model_path} не существует.")
        sys.exit(1)
  
    print("Загрузка модели...")
    model = load_model(model_path, device)
    print("Модель загружена.")
  
    print(f"Предобработка изображения {image_path}...")
    input_tensor, original_size, original_image = preprocess_image(image_path)
    input_tensor = input_tensor.to(device, dtype=torch.float32)
    print("Предобработка завершена.")
   
    print("Выполнение инференса...")
    with torch.no_grad():
        output = model(input_tensor)
    print("Инференс завершён.")

    
    print("Постобработка маски...")
    mask = postprocess_mask(output, original_size)
    print("Постобработка завершена.")
   
    print("Наложение маски на изображение...")
    overlay_image = overlay_mask_on_image(original_image, mask, color=color, alpha=alpha)
    print("Наложение завершено.")

    
    print("Визуализация результатов...")
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 3, 1)
    plt.title('Исходное изображение')
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Маска "грязи"')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Изображение с наложенной маской')
    plt.imshow(overlay_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
   
    overlay_output_path = 'vis.png'

    cv2.imwrite(overlay_output_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

    print(f"Наложенное изображение сохранено по пути: {overlay_output_path}")

if __name__ == "__main__":
    main()
