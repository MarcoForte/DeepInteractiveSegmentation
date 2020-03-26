from torch.utils.data import Dataset
import cv2
import os


def read_image(name):
    return (cv2.imread(name) / 255.0)[:, :, ::-1]


def read_gt(name):
    return cv2.imread(name, 0) / 255.0


class AlphaTestDataset(Dataset):

    def __init__(self, root_dir):
        self.alpha_dir, self.img_dir = [os.path.join(root_dir, x) for x in ['labels', 'images']]
        self.img_names = os.listdir(self.alpha_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        img_name = self.img_names[i]

        img = read_image(os.path.join(self.img_dir, img_name))
        alpha = read_gt(os.path.join(self.alpha_dir, img_name))

        return {'image': img, 'alpha': alpha, 'name': img_name[:-4]}