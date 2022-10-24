import torch.utils.data as data

from PIL import Image
import os
import os.path
from torchvision import transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes_ID = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes_ID.sort()
    class_to_idx_ID = {classes_ID[i]: i for i in range(len(classes_ID))}
    return classes_ID, class_to_idx_ID 


def make_dataset(data_root, class_to_idxID):
    images = []
    img_dir = os.path.expanduser(data_root)
    for target in sorted(os.listdir(img_dir)):
        d = os.path.join(img_dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idxID[target], class_to_idxID)
                    images.append(item)
    return images


def default_loader(path):
    return Image.open(path).convert("RGB")


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None,  
                 target_transform=None, loader=default_loader):
        classesID, class_to_idxID = find_classes(root)
        imgs = make_dataset(root, class_to_idxID)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.rgbroot = root
        # self.depthroot = depthroot
        self.imgs = imgs
        self.classesID = classesID
        self.class_to_idxID = class_to_idxID
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        tar = [None] * 3
        path_rgb, tarID, tar_dict = self.imgs[index]
        rgb_img = self.loader(path_rgb)
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
        if self.target_transform is not None:
            tarID = self.target_transform(tarID)

        return rgb_img, tarID, path_rgb, tar_dict

    def __len__(self):
        return len(self.imgs)


class PathFolder(data.Dataset):


    def __init__(self, pseudolist, transform=None, 
                 target_transform=None, loader=default_loader):
        if len(pseudolist) == 0:
            raise(RuntimeError("Found 0 images in pseudo dataset"))

        self.imgs = pseudolist
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path_rgb, tarID, confid_rate, dom_variance, weights, true_label = self.imgs[index]
        rgb_img = self.loader(path_rgb[0])
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
        if self.target_transform is not None:
            tarID = self.target_transform(tarID)

        return rgb_img, tarID, confid_rate, dom_variance, weights, true_label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = '/data2/tianjiayi/dataset/Office31/'
    dsets = {}
    source = 'amazon'
    target = 'webcam'
    dsets['source'] = ImageFolder(data_dir+source+'/images', data_transforms['train'])
    dsets['target'] = ImageFolder(data_dir+target+'/images', data_transforms['train'])
    print(len(dsets['source']))
    print(len(dsets['target']))
