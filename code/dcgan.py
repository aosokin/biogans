import numbers

import torch
import torch.nn as nn


class DCGAN_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, n_extra_layers=0, use_batch_norm=True):
        super(DCGAN_D, self).__init__()

        if isinstance(isize, numbers.Number):
            isize = (int(isize), int(isize))

        assert len(isize) == 2, "Size has to be a tuple of length 2 or a single integer"
        assert isize[0] % 2 == 0 and isize[1] % 2 == 0, "Bad image size: has to be divisible by 2 enough"

        csize = (isize[0] // 2, isize[1] // 2)
        cndf = ndf

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial.{0}-{1}.conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.{0}.LeakyReLU'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            if use_batch_norm:
                main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cndf),
                                nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}.{1}.LeakyReLU'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        # Tower
        while csize[0] > 4 and csize[1] > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            if use_batch_norm:
                main.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                                nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.LeakyReLU'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))

            assert isize[0] % 2 == 0 and isize[1] % 2 == 0, "Bad image size: has to be divisible by 2 enough"
            csize = (csize[0] // 2, csize[1] // 2)
            cndf = cndf * 2

        # Final layers
        main.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, csize, 1, 0, bias=False))
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)


class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0, red_portion=None):
        super(DCGAN_G, self).__init__()

        if isinstance(isize, numbers.Number):
            isize = (int(isize), int(isize))

        assert len(isize) == 2, "Size has to be a tuple of length 2 or a single integer"

        tisize = (isize[0], isize[1])
        cngf = ngf // 2
        while tisize[0] > 4 and tisize[1] > 4:
            assert tisize[0] % 2 == 0 and tisize[1] % 2 == 0, "Bad image size: has to be divisible by 2 enough"
            tisize = (tisize[0] // 2, tisize[1] // 2)
            cngf = cngf * 2

        is_separable = red_portion is not None
        if is_separable:
            assert isinstance(red_portion, numbers.Number) and 0.0 <= red_portion <= 1.0
            convt_name = 'convt-sep'
            convt = lambda n_ch_in, n_ch_out, size, stride, pad, bias: CnvTranspose2d_separable(n_ch_in, n_ch_out, size, stride, pad, bias=bias, red_portion=red_portion)
            conv_name = 'conv-sep'
            conv = lambda n_ch_in, n_ch_out, size, stride, pad, bias: Cnv2d_separable(n_ch_in, n_ch_out, size, stride, pad, bias=bias)
        else:
            convt_name = 'convt'
            convt = lambda n_ch_in, n_ch_out, size, stride, pad, bias: nn.ConvTranspose2d(n_ch_in, n_ch_out, size, stride, pad, bias=bias)
            conv_name = 'conv'
            conv = lambda n_ch_in, n_ch_out, size, stride, pad, bias: nn.Conv2d(n_ch_in, n_ch_out, size, stride, pad, bias=bias)

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial.{0}-{1}.{2}'.format(nz, cngf, convt_name),
                        convt(nz, cngf, tisize, 1, 0, bias=False))
        main.add_module('initial.{0}.batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial.{0}.ReLU'.format(cngf),
                        nn.ReLU(True))

        # Tower
        csize = tisize[0]
        while csize < isize[0] // 2:
            main.add_module('pyramid.{0}-{1}.{2}'.format(cngf, cngf // 2, convt_name),
                            convt(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid.{0}.ReLU'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.{2}'.format(t, cngf, conv_name),
                            conv(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}.{1}.ReLU'.format(t, cngf),
                            nn.ReLU(True))

        # Final layers
        main.add_module('final.{0}-{1}.{2}'.format(cngf, nc, convt_name),
                        convt(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final.{0}.tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


########## Layers for separable DCGAN generator ###########################################
class CnvTranspose2d_separable(nn.Module):
    def __init__(self, n_input_ch, n_output_ch, kernel_size, stride, padding, bias=False, red_portion=0.5):
        super(CnvTranspose2d_separable, self).__init__()

        self.n_input_ch = n_input_ch
        self.n_input_ch_red = int(n_input_ch * red_portion)

        self.n_output_ch = n_output_ch
        self.n_output_ch_red = int(n_output_ch * red_portion)
        self.n_output_ch_green = n_output_ch - self.n_output_ch_red

        self.convt_half = nn.ConvTranspose2d(self.n_input_ch_red, self.n_output_ch_red,
                                             kernel_size, stride, padding, bias=bias)
        self.convt_all = nn.ConvTranspose2d(self.n_input_ch, self.n_output_ch_green,
                                            kernel_size, stride, padding, bias=bias)

    def forward(self, input):
        first_half = input[:, :self.n_input_ch_red, :, :]
        first_half_conv = self.convt_half(first_half)
        full_conv = self.convt_all(input)
        all_conv = torch.cat((first_half_conv, full_conv), 1)
        return all_conv


class Cnv2d_separable(nn.Module):
    def __init__(self, n_input_ch, n_output_ch, kernel_size, stride, padding, bias=False, red_portion=0.5):
        super(Cnv2d_separable, self).__init__()

        self.n_input_ch = n_input_ch
        self.n_input_ch_red = int(n_input_ch * red_portion)

        self.n_output_ch = n_output_ch
        self.n_output_ch_red = int(n_output_ch * red_portion)
        self.n_output_ch_green = n_output_ch - self.n_output_ch_red

        self.conv_half = nn.Conv2d(self.n_input_ch_red, self.n_output_ch_red,
                                   kernel_size, stride, padding, bias=bias)
        self.conv_all = nn.Conv2d(self.n_input_ch, self.n_output_ch_green,
                                  kernel_size, stride, padding, bias=bias)

    def forward(self, input):
        first_half = input[:, :self.n_input_ch_red, :, :]
        first_half_conv = self.conv_half(first_half)
        full_conv = self.conv_all(input)
        all_conv = torch.cat((first_half_conv, full_conv), 1)
        return all_conv
