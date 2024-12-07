import os
import sys
import cv2
import json
import base64
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision import transforms


# Определение модели
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


# Извлечение признаков
def local_binary_pattern_custom(image, P=8, R=1):
    lbp = np.zeros_like(image, dtype=np.uint8)
    neighbors = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1)
    ]

    for i, (dy, dx) in enumerate(neighbors):
        shifted = np.roll(np.roll(image, dy, axis=0), dx, axis=1)
        lbp |= ((shifted >= image) << (P - 1 - i)).astype(np.uint8)

    return lbp


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
    lbp = local_binary_pattern_custom(gray)
    lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min())

    features = np.stack([laplacian, sobel, lbp], axis=-1)
    return features


# Загрузка модели
def load_model(model_path, device):
    model = UNet(n_channels=6, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Предобработка изображения
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Извлечение признаков
    features = extract_aberration_features(image_rgb)

    # Объединение каналов
    combined = np.concatenate([image_rgb, features], axis=-1)

    # Изменение размера
    combined_resized = cv2.resize(combined, (256, 256))

    # Преобразование в тензор
    combined_tensor = torch.from_numpy(np.transpose(combined_resized, (2, 0, 1))).float().unsqueeze(0)
    return combined_tensor, image.shape[:2]


def postprocess_mask(mask_tensor, original_size, gamma=2.0, focus_min=0.90, focus_max=1.0):
    mask = torch.sigmoid(mask_tensor) 
    mask = mask.squeeze().cpu().numpy()  

    mask = np.clip(mask, focus_min, focus_max)  
    mask = (mask - focus_min) / (focus_max - focus_min)  

    mask = np.power(mask, gamma)

    mask = (mask * 255).astype(np.uint8)

    mask_resized = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    return mask_resized


# Кодирование маски в Base64
def encode_mask_to_base64(mask):
    _, encoded_img = cv2.imencode(".png", mask)
    return base64.b64encode(encoded_img).decode('utf-8')


# Основная функция
def main():
    if len(sys.argv) != 3:
        sys.exit(1)

    input_dir = sys.argv[1]
    output_json = sys.argv[2]
    model_path = "models3/BaDmodel_45.pth"
    device = torch.device('cpu')

    model = load_model(model_path, device)

    results = {}
    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(('.jpg', '.png')):
            continue

        image_path = os.path.join(input_dir, filename)
        input_tensor, original_size = preprocess_image(image_path)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)

        mask = postprocess_mask(output, original_size)
        encoded_mask = encode_mask_to_base64(mask)
        results[filename] = encoded_mask

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_json}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    delta = time.time()-start_time
    print(delta/60)