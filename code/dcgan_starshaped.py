import numbers
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.utils.data
import torchvision.transforms as transforms

import dcgan
from custom_dataloaders import ImageFolderWithCache, read_all_images


class DatasetConditionalCached(object):
    def __init__(self, data_path, do_random_flips=False, num_workers=0,
                 normalization=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
        self.classes = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
        self.classes = sorted(self.classes)
        datasets = OrderedDict()
        for cl in self.classes:
            print('Preparing class', cl)
            class_path = os.path.join(data_path, cl)
            image_cache = read_all_images(class_path, num_workers)
            datasets[cl] = ImageFolderWithCache(class_path, image_cache, do_random_flips=do_random_flips,
                                                normalization=normalization)
        print('')
        self.class_datasets = datasets


class DataLoaderConditionalCached(object):
    def __init__(self, dataset, batch_size=64, shuffle=True):
        self.class_loaders = OrderedDict()

        for key, value in dataset.class_datasets.items():
            self.class_loaders[key] = torch.utils.data.DataLoader(value, batch_size=batch_size,
                                                                  shuffle=shuffle, num_workers=0, drop_last=True)

    def __iter__(self):
        iters = [iter(l) for cl, l in self.class_loaders.items()]
        for i_step in range(len(self)):
            data_list = [next(i) for i in iters]
            num_items = len(data_list[0])
            data_th = [None] * num_items
            for i_item in range(num_items):
                ii = [l[i_item] for l in data_list]
                data_th[i_item] = torch.stack(ii, 0)
            yield data_th

    def __len__(self):
        # consider minimum of the class sizes as the final class size
        return min([len(value) for key, value in self.class_loaders.items()])


class DCGAN_D_starShaped(nn.Module):
    def __init__(self, isize, nz, nc, ndf, n_classes, n_extra_layers=0, use_batch_norm=True):
        super(DCGAN_D_starShaped, self).__init__()

        self.n_classes = n_classes
        main = nn.ModuleList()
        for i_c in range(n_classes):
            D_this_class = dcgan.DCGAN_D(isize, nz, nc, ndf, n_extra_layers=n_extra_layers, use_batch_norm=use_batch_norm)
            main.append(D_this_class)
        self.main = main

    def forward(self, input):
        assert(input.size(0) == self.n_classes)
        output = [None] * self.n_classes
        for i_class in range(self.n_classes):
            output[i_class] = self.main[i_class](input[i_class])
        output = torch.stack(output, 0)
        return output


class DCGAN_G_starShaped_from_multi_channel(nn.Module):
    def __init__(self, n_classes, netG):
        super(DCGAN_G_starShaped_from_multi_channel, self).__init__()

        self.n_classes = n_classes
        self.G = netG

    def forward(self, input, i_class=None):
        images = self.G(input)
        # reorganizing inputs to feed the discriminator for the star-shaped models
        assert (images.size(1) == self.n_classes + 1)
        if i_class is None:
            output = [None] * self.n_classes
            for i_class in range(self.n_classes):
                output[i_class] = torch.stack([images[:,0,:,:], images[:,i_class + 1,:,:]], 1)
            output = torch.stack(output, 0)
        else:
            output = torch.stack([images[:, 0, :, :], images[:, i_class + 1, :, :]], 1)
        return output


###############################################################################
class CnvTranspose2d_starShape(nn.Module):
    def __init__(self, n_input_ch, n_output_ch, kernel_size, stride, padding, n_classes,
                 bias=False, red_portion_in=0.5, red_portion_out=0.5,
                 use_batch_norm=True, activation=None):
        super(CnvTranspose2d_starShape, self).__init__()

        self.n_input_ch = n_input_ch
        self.n_input_ch_red = int(n_input_ch * red_portion_in)

        self.n_output_ch = n_output_ch
        self.n_output_ch_red = int(n_output_ch * red_portion_out)
        self.n_output_ch_green = n_output_ch - self.n_output_ch_red

        self.n_classes = n_classes
        self.use_batch_norm = use_batch_norm

        self.convt_red = nn.ConvTranspose2d(self.n_input_ch_red, self.n_output_ch_red,
                                            kernel_size, stride, padding, bias=bias)

        if self.use_batch_norm:
            self.batchnorm_red = nn.BatchNorm2d(self.n_output_ch_red)

        self.convt_green = nn.ModuleList()
        if self.use_batch_norm:
            self.batchnorm_green = nn.ModuleList()
        for i_c in range(n_classes):
            self.convt_green.append(nn.ConvTranspose2d(self.n_input_ch, self.n_output_ch_green,
                                                       kernel_size, stride, padding, bias=bias))
            if self.use_batch_norm:
                self.batchnorm_green.append(nn.BatchNorm2d(self.n_output_ch_green))

        self.activation = activation

    def forward(self, input_red, input_green, i_class=None):
        output_red = self.convt_red(input_red)
        if self.use_batch_norm:
            output_red = self.batchnorm_red(output_red)
        if self.activation:
            output_red = self.activation(output_red)

        if i_class is None:
            output_green = [None] * self.n_classes
            for i_class in range(self.n_classes):
                input_this_green = torch.cat([input_red, input_green[i_class]], 1)
                output_green[i_class] = self.convt_green[i_class](input_this_green)
                if self.use_batch_norm:
                    output_green[i_class] = self.batchnorm_green[i_class](output_green[i_class])
                if self.activation:
                    output_green[i_class] = self.activation(output_green[i_class])
        else:
            input = torch.cat([input_red, input_green], 1)
            output_green = self.convt_green[i_class](input)
            if self.use_batch_norm:
                output_green = self.batchnorm_green[i_class](output_green)
            if self.activation:
                output_green = self.activation(output_green)

        return output_red, output_green


class DCGAN_G_starShaped(nn.Module):
    def __init__(self, isize, nz, nc, ngf, n_classes, red_portion=0.5):
        super(DCGAN_G_starShaped, self).__init__()

        if isinstance(isize, numbers.Number):
            isize = (int(isize), int(isize))

        assert len(isize) == 2, "Size has to be a tuple of length 2 or a single integer"

        tisize = (isize[0], isize[1])
        cngf = ngf // 2
        while tisize[0] > 4 and tisize[1] > 4:
            assert tisize[0] % 2 == 0 and tisize[1] % 2 == 0, "Bad image size: has to be divisible by 2 enough"
            tisize = (tisize[0] // 2, tisize[1] // 2)
            cngf = cngf * 2

        # initial block
        main = nn.ModuleList()
        main.append(CnvTranspose2d_starShape(nz, cngf, tisize, 1, 0, n_classes, bias=False,
                                             red_portion_in=red_portion, red_portion_out=red_portion,
                                             use_batch_norm=True, activation=nn.ReLU(inplace=True)))


        # convt tower
        csize = tisize[0]
        while csize < isize[0] // 2:
            main.append(CnvTranspose2d_starShape(cngf, cngf // 2, 4, 2, 1, n_classes, bias=False,
                                                 red_portion_in=red_portion, red_portion_out=red_portion,
                                                 use_batch_norm=True, activation=nn.ReLU(inplace=True)))
            cngf = cngf // 2
            csize = csize * 2

        # final block
        main.append(CnvTranspose2d_starShape(cngf, nc, 4, 2, 1, n_classes, bias=False,
                                             red_portion_in=0.5, red_portion_out=0.5,
                                             use_batch_norm=False, activation=nn.Tanh()))

        self.nc = nc
        self.nz = nz
        self.n_classes = n_classes
        self.nz_red = int(self.nz * red_portion)
        self.nz_green = self.nz - self.nz_red

        self.main = main

    def forward(self, input, i_class=None):
        output_red = input[:, :self.nz_red, :, :]
        output_green = input[:, self.nz_red:, :, :]

        if i_class is None:
            output_green = [output_green] * self.n_classes

            for group in self.main:
                output_red, output_green = group(output_red, output_green)

            output_images = [None] * self.n_classes
            for i_class in range(self.n_classes):
                output_images[i_class] = torch.cat([output_red, output_green[i_class]], 1)
            output = torch.stack(output_images, 0)
        else:
            for group in self.main:
                output_red, output_green = group(output_red, output_green, i_class=i_class)
            output = torch.cat([output_red, output_green], 1)
        return output


###############################################################################


class DCGAN_G_independent(nn.Module):
    def __init__(self, isize, nz, nc, ngf, n_classes, red_portion=None):
        super( DCGAN_G_independent, self).__init__()

        self.n_classes = n_classes
        main = nn.ModuleList()
        for i_c in range(n_classes):
            G_this_class = dcgan.DCGAN_G(isize, nz, nc, ngf, red_portion=red_portion)
            main.append(G_this_class)
        self.main = main

    def forward(self, input, i_class=None):
        if i_class is None:
            output_images = [None] * self.n_classes
            for i_class in range(self.n_classes):
                output_images[i_class] = self.main[i_class](input)
            output = torch.stack(output_images, 0)
        else:
            output = self.main[i_class](input)
        return output