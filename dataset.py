from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import config
from torchvision.utils import save_image

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        print(self.list_files)
    
    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]  
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        # Trasform images with augmentatinons lib to resize and normalize
        # Images most be of size 256x256
        augmentations = config.both_transform(image=input_image, image0=target_image)

        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        # Augmantate input image and target image
        input_image = config.transform_only_input(image = input_image)['image']
        target_image = config.transform_only_mask(image = target_image)['image']

        return input_image, target_image

if __name__ == "__main__":
    dataset = MapDataset("data/maps/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()