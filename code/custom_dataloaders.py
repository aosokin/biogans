import os
from torchvision.datasets.folder import default_loader, find_classes, make_dataset, IMG_EXTENSIONS
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import torch
import random
from tqdm import tqdm

from utils import parallel_process


def read_image_for_pytorch(image_file_name):
    img = default_loader(image_file_name)

    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if img.mode == 'YCbCr':
        nchannel = 3
    else:
        nchannel = len(img.mode)

    # convert to numpy array
    img = np.array(img.getdata()).reshape(img.size[1], img.size[0], nchannel)

    # permute dimensions
    img = np.transpose(img, (2, 0, 1)).copy()
    return img


def read_all_images(root, num_workers=4):
    classes, class_to_idx = find_classes(root)
    dataset = make_dataset(root, class_to_idx)
    if len(dataset) == 0:
        raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n" +
                            "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    num_images = len(dataset)
    paths = [dataset[i_image][0] for i_image in range(num_images)]

    print("Reading {0} images with {1} workers".format(num_images, num_workers))
    if num_workers > 1:
        images = parallel_process(paths, read_image_for_pytorch, n_jobs=num_workers)
    else:
        images = []
        for p in tqdm(paths):
            images.append(read_image_for_pytorch(p))

    image_cache = {}
    for i, image in enumerate(images):
        path, target = dataset[i]
        image_cache[path] = image
    return image_cache


class ImageFolderWithCache(data.Dataset):

    def __init__(self, data_path, image_cache, do_random_flips=False,
                 normalization=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
        classes, class_to_idx = find_classes(data_path)
        imgs = make_dataset(data_path, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + data_path + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = data_path
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.normalization = normalization
        self.do_random_flips = do_random_flips
        self.image_cache = image_cache

    def read_image_with_cache(self, image_file_name):

        if image_file_name not in self.image_cache:
            self.image_cache[image_file_name] = read_image_for_pytorch(image_file_name)

        return self.image_cache[image_file_name]

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.read_image_with_cache(path)

        # pytorch does not have reverse indexing, so I;m using numpy for that
        if self.do_random_flips:
            if random.random() < 0.5:
                img = img[:, ::-1, :]
            if random.random() < 0.5:
                img = img[:, :, ::-1]
            img = img.copy()

        # convert to torch tensor manually (torchvision.transforms.ToTensor is buggy)
        img = torch.from_numpy(img.astype(np.float32)) / 255.0
        assert(img.size(0) == 3)

        if self.normalization is not None:
            img_norm = []
            for i_c in range(img.size(0)):
                img_norm.append(self.normalization(img[i_c].unsqueeze(0)))
            img = torch.cat(img_norm, 0).contiguous()

        return img, target

    def __len__(self):
        return len(self.imgs)


##################################
class CompositeImageFolder(data.Dataset):
    """
    Like ImageFolder, but creates a multi channel image with n_classes channels.

    The red channel is always the first channel in the image.
    """

    def __init__(self, data_path, nn_dict, image_cache, do_random_flips=False,
                 normalization=None):
        classes, class_to_idx = find_classes(data_path)
        imgs = make_dataset(data_path, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + data_path + "\n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.data_path = data_path
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.normalization = normalization
        self.do_random_flips = do_random_flips
        self.nn_dict = nn_dict
        self.image_cache = image_cache

    def read_image_with_cache(self, image_file_name):

        if image_file_name not in self.image_cache:
            self.image_cache[image_file_name] = read_image_for_pytorch(image_file_name)

        return self.image_cache[image_file_name]

    def __getitem__(self, index):

        path, target = self.imgs[index]

        _img = self.read_image_with_cache(path)

        image_size = (_img.shape[1], _img.shape[2])
        total_channels = len(self.classes) + 1
        img = np.zeros((total_channels, image_size[0], image_size[1]))

        # We use the red to create a binary mask
        binary_mask = _img[0, :, :] > 0

        # We move the green of the current image into the right channel index
        img[target + 1, :, :] = _img[1, :, :] * binary_mask
        img[0, :, :] = _img[0, :, :] * binary_mask

        # Load the channels into the image.
        for c in self.classes:
            c_idx = self.class_to_idx[c]
            if c_idx == target:
                continue

            file_name = os.path.basename(path)
            nn_file = np.random.choice(self.nn_dict[file_name][c])

            _nn_img = self.read_image_with_cache(os.path.join(self.data_path, c, nn_file))

            # Move the green of that image into the correct channel of the image.
            img[c_idx + 1, :, :] = _nn_img[1, :, :] * binary_mask

        # pytorch does not have reverse indexing, so I;m using numpy for that
        if self.do_random_flips:
            if random.random() < 0.5:
                img = img[:, ::-1, :]
            if random.random() < 0.5:
                img = img[:, :, ::-1]
            img = img.copy()

        # convert to torch tensor manually (torchvision.transforms.ToTensor is buggy)
        img = torch.from_numpy(img.astype(np.float32)) / 255.0
        assert(img.size(0) == total_channels)

        if self.normalization is not None:
            img_norm = []
            for i_c in range(img.size(0)):
                img_norm.append(self.normalization(img[i_c].unsqueeze(0)))
            img = torch.cat(img_norm, 0).contiguous()

        return img, target

    def __len__(self):
        return len(self.imgs)
