# dataset.py
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

class YOLODataset(Dataset):
    def __init__(self, img_dir, labels_dir, transform=None):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_file)[0] + '.txt')
        
        image = Image.open(img_path).convert("RGB")
        boxes = self.read_yolo_label(label_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, boxes

    def read_yolo_label(self, label_path):
        boxes = []
        with open(label_path, 'r') as file:
            for line in file.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                boxes.append([class_id, x_center, y_center, width, height])
        return torch.tensor(boxes, dtype=torch.float32)

# 이미지 변환 정의
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# 커스텀 collate_fn 정의
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets
