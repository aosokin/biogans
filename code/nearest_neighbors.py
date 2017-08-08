import os, sys
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data
from torchvision.transforms import  ToTensor

file_path = os.path.dirname(__file__)
data_path = os.path.join(file_path, '..', 'data')
query_datasets = [os.path.join(data_path, 'LIN_Normalized_WT_size-48-80_train'),
                  os.path.join(data_path, 'LIN_Normalized_WT_size-48-80_test')]
search_dataset = os.path.join(data_path, 'LIN_Normalized_WT_size-48-80_train')
classes_of_interest = ['Alp14', 'Arp3', 'Cki2', 'Mkh1', 'Sid2', 'Tea1']

use_cuda = True
num_neighbors = 5

result_file = os.path.join(data_path, 'red_neighbors_LIN_Normalized_WT_size-48-80_all_to_train.pth')


def get_files_in_folder(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files.sort()
    return files

def load_image_to_th(path, num_channels=None):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            transform = ToTensor()
            im_th = transform(img.convert('RGB'))
            if num_channels is None:
                return im_th
            else:
                return im_th[:num_channels, :, :].contiguous()

def compute_L2_dists(query_th, ims_th):
    assert(query_th.size() == ims_th.size()[1:])
    query_th = query_th.view(-1)
    ims_th = ims_th.view(-1, query_th.size(0))
    return torch.sum((query_th.unsqueeze(0) - ims_th)**2, 1)

def read_search_images(search_dataset=search_dataset, classes_of_interest=classes_of_interest, num_channels=None, use_cuda=use_cuda):
    # read all search_images
    print('Reading all search images')
    sys.stdout.flush()

    search_images = {}
    search_image_names = {}
    for cl_search in classes_of_interest:
        print('Reading class', cl_search)
        sys.stdout.flush()

        # make all the images neighbors
        search_path = os.path.join(search_dataset, cl_search)
        ims = get_files_in_folder(search_path)
        if not ims:
            search_path = os.path.join(search_path, cl_search)
            ims = get_files_in_folder(search_path)
            if not ims:
                raise('Failed to load images of class {0}'.format(cl_search))

        ims_th = []
        for im in tqdm(ims):
            im_th = load_image_to_th(os.path.join(search_path, im), num_channels=num_channels)
            ims_th.append(im_th)
        ims_th = torch.stack(ims_th, 0)
        if use_cuda:
            ims_th = ims_th.cuda()

        search_images[cl_search] = ims_th
        search_image_names[cl_search] = ims
    return search_images, search_image_names


def find_neighbors(im, image_set, image_names, num_neighbors=num_neighbors):
    # compute the L2 distances
    dists = compute_L2_dists(im, image_set)
    # sort in the order of increasing distance
    sorted, indices = torch.sort(dists, dim=0, descending=False)
    indices = indices.cpu()
    # pick the nearest neighbors
    nn_names = [image_names[i] for i in indices[:num_neighbors]]
    return nn_names, indices[:num_neighbors]


def find_neighbors_to_dataset(query_dataset, search_dataset, num_channels_to_use=None, use_cuda=use_cuda):
    # read all search_images
    search_images, search_image_names = read_search_images(search_dataset=search_dataset, num_channels=num_channels_to_use)

    # search for neighbors
    neighbors = {}
    for cl in classes_of_interest:
        print('\nSearching in class', cl)
        query_path = os.path.join(query_dataset, cl)
        query_images = get_files_in_folder(query_path)
        for im_name in tqdm(query_images):
            im = load_image_to_th(os.path.join(query_path, im_name), num_channels=num_channels_to_use)
            if use_cuda:
                im = im.cuda()

            im_dict = {}
            for cl_search in classes_of_interest:
                im_dict[cl_search], _ = find_neighbors(im, search_images[cl_search], search_image_names[cl_search])
            neighbors[im_name] = im_dict
    return neighbors


if __name__ == "__main__":
    neighbors = {}
    for query_dataset in query_datasets:
        # search neighbors based only on the red channel
        neighbors.update(find_neighbors_to_dataset(query_dataset, search_dataset, num_channels_to_use=1))
    torch.save(neighbors, result_file)
