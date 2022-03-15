import numpy as np
import torchvision.transforms
from torch.utils.data import Dataset
import os
import cv2
from torch.utils.data import DataLoader

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 320


class SegmentationDataSet(Dataset):

    def __init__( self, image_path, label_path):
        self.img_path = image_path
        self.label_path = label_path
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        self.img_names = os.listdir(self.img_path)


    def __getitem__(self, idx):
        img_path = self.img_path + "/" + self.img_names[idx]
        img = cv2.imread(img_path)
        img = img.astype(np.float32)
        img = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation = cv2.INTER_AREA)

        norm = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
        img = cv2.normalize(img, norm, 0, 255, cv2.NORM_MINMAX)

        img = self.transform(img)


        label_path = self.label_path+ "/" + self.img_names[idx].split(".")[0]+"_mask.jpg"
        label = cv2.imread(label_path, 0)
        label = label.astype(np.float32)
        label[label == 255.0] = 1.0
        label = cv2.resize(label, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_AREA)
        label[label >= 0.5] = 1.0
        label[label < 0.5] = 0.0
        label = self.transform(label)

        return img, label



    def __len__(self):
        return len(self.img_names)
#
#
#
if __name__ == '__main__':

    train_data = SegmentationDataSet("data/train", "data/train_masks")
    train_loader = DataLoader(dataset = train_data, batch_size = 1, shuffle = True, num_workers = 0, drop_last = False)

    for d in train_loader:
        x,y = d

        break


