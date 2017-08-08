import sys
import math
from time import time as time

import torch
from torch.autograd import Variable
import torch.utils.data
import torchvision.utils as vutils

from utils import pad_channels


class GanImageloader():
    def __init__(self):
        self.i_epoch = 0
        self.last_images = None

    def return_iterator(self, dataloader, is_cuda, num_channels, return_labels=False, num_passes=None):
        self.i_epoch = 0
        self.last_images = None
        while num_passes is None or self.i_epoch < num_passes:
            for data in dataloader:
                real_images, labels = data
                if real_images.dim() == 4:
                    # for regular image datasets
                    real_images = real_images[:, :num_channels, :, :]
                elif real_images.dim() == 5:
                    # for star-shaped datasets
                    real_images = real_images[:, :, :num_channels, :, :]
                else:
                    raise RuntimeError("Unknown dimension of image data: {0}".format(real_images.dim()))
                if is_cuda:
                    real_images = real_images.cuda()
                    labels = labels.cuda()
                self.last_images = real_images
                real_images = Variable(real_images)
                labels = Variable(labels)
                if return_labels:
                    yield real_images, labels
                else:
                    yield real_images
            self.i_epoch += 1


class GAN_base():
    def __init__(self, netG, netD, optimizerD, optimizerG, opt):
        self.netD, self.netG = netD, netG
        self.optimizerD, self.optimizerG = optimizerD, optimizerG
        self.is_cuda = opt.cuda
        if self.is_cuda:
            self.netD.cuda()
            self.netG.cuda()

    def compute_disc_score(self, data_a, data_b):
        raise NotImplementedError
        errD = None
        return errG

    def compute_gen_score(self, data):
        raise NotImplementedError
        errD = None
        return errD

    def train_D_one_step(self, iterator_a, iterator_b):
        self.netD.zero_grad()
        for p in self.netD.parameters():
            p.requires_grad = True  # to avoid computation

        # get data and scores
        data_a = next(iterator_a)
        data_b = next(iterator_b)
        errD = self.compute_disc_score(data_a.detach(), data_b.detach())
        errD = errD.mean()
        errD.backward()
        self.optimizerD.step()
        return errD.data[0], data_a, data_b

    def train_G_one_step(self, iterator_fake, fake_images=None):
        self.netG.zero_grad()
        for p in self.netD.parameters():
            p.requires_grad = False  # to avoid computation

        if fake_images is None:
            fake_images = next(iterator_fake)
        errG = self.compute_gen_score(fake_images)

        errG.backward()
        self.optimizerG.step()
        return errG.data[0], fake_images

    def train_one_step(self, iterator_data, iterator_fake, num_disc_iters=1, i_iter=None):
        fake_images, errD, errG = None, None, None
        # Update D network
        for i in range(num_disc_iters):
            errD, real_data, fake_images = self.train_D_one_step(iterator_data, iterator_fake)
        # Update G network
        errG, fake_images = self.train_G_one_step(iterator_fake, fake_images)
        return errD, errG

    def train(self, dataloader, opt):
        netD, netG = self.netD, self.netG

        # create fixed noise for evaluation
        fixed_noise = self.generate_fixed_noise(opt)

        # move everything on a GPU
        if self.is_cuda:
            netD.cuda()
            netG.cuda()
            fixed_noise = fixed_noise.cuda()

        # iterators
        data_looper = GanImageloader()
        iterator_data = data_looper.return_iterator(dataloader, self.is_cuda, opt.nc, return_labels=False)
        iterator_fake = self.fake_data_generator(opt.batch_size, opt.nz)

        # main loop
        t_start = time()
        for i_iter in range(opt.num_iter):
            t_iter_start = time()

            netD.train()
            netG.train()
            errD, errG = self.train_one_step(iterator_data, iterator_fake,
                                             num_disc_iters=opt.num_disc_iters, i_iter=i_iter)

            cur_time = time()
            print('[%d/%d][epoch %d] Loss_D: %f; Loss_G: %f; Batch time: %fs; Full time: %fh'
                  % (i_iter, opt.num_iter, data_looper.i_epoch,
                     errD, errG,
                     cur_time - t_iter_start, (cur_time - t_start) / 3600))
            sys.stdout.flush()

            # do checkpointing
            model_saved = False
            if opt.save_iter is not None and (i_iter + 1) % opt.save_iter == 0:
                model_saved = True
            if (i_iter + 1) in [10, 100, 1000, 2000, 3000, 5000, 10000, 30000, 50000, 100000, 300000, 500000]:
                model_saved = True

            if model_saved:
                self.save_models(i_iter + 1, opt)
            if model_saved or (i_iter + 1) % opt.image_iter == 0:
                self.save_samples(data_looper.last_images, fixed_noise, i_iter + 1, opt)

        # save the final results of training
        self.save_samples(data_looper.last_images, fixed_noise, opt.num_iter, opt)
        if opt.save_iter is not None:
            self.save_models(opt.num_iter, opt)

    def gen_latent_noise(self, batch_size, nz):
        th = torch.cuda if self.is_cuda else torch
        return th.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)

    def gen_fake_data(self, batch_size, nz, noise=None):
        if noise is None:
            noise = Variable(self.gen_latent_noise(batch_size, nz))
        return self.netG(noise)

    def fake_data_generator(self, batch_size, nz):
        while True:
            yield self.gen_fake_data(batch_size, nz)

    def generate_fixed_noise(self, opt):
        if opt.fixed_noise_file:
            print('Loading fixed noise from {0}'.format(opt.fixed_noise_file))
            fixed_noise = torch.load(opt.fixed_noise_file)
        else:
            fixed_noise = self.gen_latent_noise(opt.batch_size, opt.nz)
            if opt.separable_gen:
                # do groups of 8 samples with the same red
                group_size = 8
                n_red_noise = int(opt.g_input_size * opt.red_portion)
                for i_group in range(math.ceil(opt.batch_size / float(group_size))):
                    group_start = i_group * group_size
                    for i_element in range(1, min(group_size, opt.batch_size - group_start)):
                        fixed_noise[group_start + i_element, :n_red_noise] = fixed_noise[group_start, :n_red_noise]
            # torch.save(fixed_noise.cpu(), opt.fixed_noise_file)
        return Variable(fixed_noise, volatile=True)

    def save_models(self, i_iter, opt):
        torch.save(self.netG.cpu().state_dict(), '{0}/netG_iter_{1}.pth'.format(opt.experiment, i_iter))
        torch.save(self.netD.cpu().state_dict(), '{0}/netD_iter_{1}.pth'.format(opt.experiment, i_iter))
        if self.is_cuda:
            self.netD.cuda()
            self.netG.cuda()

    def save_images(self, images, file_prefix, opt):
        if images.dim() == 4:
            if images.size(1) == opt.n_classes + 1:
                for i_class, class_name in enumerate(opt.class_names):
                    images_class = [images[:, 0, :, :], images[:, i_class + 1, :, :]]
                    images_class = torch.stack(images_class, 1)
                    images_class = pad_channels(images_class, 3)
                    vutils.save_image(images_class * opt.std_val + opt.mean_val,
                                      '{0}_{1}.png'.format(file_prefix, class_name))
            else:
                images = pad_channels(images, 3)
                vutils.save_image(images * opt.std_val + opt.mean_val,
                                  '{0}.png'.format(file_prefix))
        elif images.dim() == 5:
            assert (images.size(0) == len(opt.class_names))
            for i_class, class_name in enumerate(opt.class_names):
                class_images = pad_channels(images[i_class], 3)
                vutils.save_image(class_images * opt.std_val + opt.mean_val,
                                  '{0}_{1}.png'.format(file_prefix, class_name))
        else:
            raise RuntimeError("Unknown dimension of image data: {0}".format(images.dim()))

    def save_samples(self, real_images, fixed_noise, i_iter, opt):
        if real_images is not None:
            self.save_images(real_images, '{0}/{1}'.format(opt.experiment, opt.prefix_real_samples), opt)

        # switch to the evaluation mode for generation: affects dropout and batchnorm
        self.netG.eval()
        if self.is_cuda:
            fixed_noise = fixed_noise.cuda()
        fake_images = self.gen_fake_data(None, None, fixed_noise)
        fake_images = fake_images.data
        if i_iter is not None:
            file_name_fake_images = '{0}/{1}_{2}'.format(opt.experiment, opt.prefix_fake_samples, i_iter)
        else:
            file_name_fake_images = '{0}/{1}'.format(opt.experiment, opt.prefix_fake_samples)
        self.save_images(fake_images, file_name_fake_images, opt)

    def train_disc_fake_vs_data(self, dataloader, opt):
        # create iterators
        data_looper = GanImageloader()
        iterator_data = data_looper.return_iterator(dataloader, self.is_cuda, opt.nc, return_labels=False)
        iterator_fake = self.fake_data_generator(opt.batch_size, opt.nz)

        self.train_disc_iterators(iterator_data, iterator_fake, opt)

    def train_disc_data_vs_data(self, dataloader_a, dataloader_b, opt):
        # create iterators
        data_looper_a = GanImageloader()
        iterator_a = data_looper_a.return_iterator(dataloader_a, self.is_cuda, opt.nc, return_labels=False)
        data_looper_b = GanImageloader()
        iterator_b = data_looper_b.return_iterator(dataloader_b, self.is_cuda, opt.nc, return_labels=False)

        self.train_disc_iterators(iterator_a, iterator_b, opt)

    def train_disc_iterators(self, iterator_a, iterator_b, opt):
        # move everything on a GPU
        if self.is_cuda:
            self.netD.cuda()
            self.netG.cuda()

        # main loop
        t_start = time()
        for i_iter in range(opt.num_iter):
            t_iter_start = time()

            self.netG.eval()
            self.netD.train()
            errD, _, _ = self.train_D_one_step(iterator_a, iterator_b)

            cur_time = time()
            print('[%d/%d] Loss_D: %f; Batch time: %fs; Full time: %fh'
                  % (i_iter, opt.num_iter,
                     errD, cur_time - t_iter_start, (cur_time - t_start) / 3600))

        self.netD.eval()

    def compute_dics_score_fake_vs_data(self, dataloader, opt):
        # create iterators
        data_looper = GanImageloader()
        iterator_data = data_looper.return_iterator(dataloader, self.is_cuda, opt.nc, return_labels=False, num_passes=5)
        iterator_fake = self.fake_data_generator(opt.batch_size, opt.nz)

        # loop over data
        score = 0.0
        num_batches = 0
        for data_real in iterator_data:
            data_fake = next(iterator_fake)
            data_fake.detach_()
            data_real.detach_()
            data_fake.volatile = True
            data_real.volatile = True
            score_var = self.compute_disc_score(data_real, data_fake)
            score += score_var
            num_batches += 1

        print('Evaluated on {0} batches'.format(num_batches))
        return score.data / num_batches

    def compute_dics_score_data_vs_data(self, dataloader_a, dataloader_b, opt):
        # create iterators
        data_looper_a = GanImageloader()
        iterator_a = data_looper_a.return_iterator(dataloader_a, self.is_cuda, opt.nc, return_labels=False, num_passes=5)
        data_looper_b = GanImageloader()
        iterator_b = data_looper_b.return_iterator(dataloader_b, self.is_cuda, opt.nc, return_labels=False, num_passes=5)

        # loop over data
        score = 0.0
        num_batches = 0
        data_a = next(iterator_a, None)
        data_b = next(iterator_b, None)
        while data_a is not None and data_b is not None:
            data_a.detach_()
            data_b.detach_()
            data_a.volatile = True
            data_b.volatile = True
            score_var = self.compute_disc_score(data_a, data_b)
            assert(score_var.numel() == 1)
            score += score_var
            num_batches += 1
            data_a = next(iterator_a, None)
            data_b = next(iterator_b, None)

        print('Evaluated on {0} batches'.format(num_batches))
        return score.data / num_batches


class GAN(GAN_base):
    def __init__(self, netG, netD, optimizerD, optimizerG, opt):
        GAN_base.__init__(self, netG, netD, optimizerD, optimizerG, opt)

        # criterion for training
        self.criterion = torch.nn.BCEWithLogitsLoss(size_average=True)
        self.real_label = 1
        self.fake_label = 0
        self.generator_label = 1  # fake labels are real for generator cost

    def compute_disc_score(self, data_a, data_b):
        th = torch.cuda if self.is_cuda else torch

        scores_a = self.netD(data_a)
        scores_b = self.netD(data_b)

        if scores_a.dim() == 1:
            labels_a = Variable(th.FloatTensor(scores_a.size(0)).fill_(self.real_label))
            errD_a = self.criterion(scores_a, labels_a)

            labels_b = Variable(th.FloatTensor(scores_b.size(0)).fill_(self.fake_label))
            errD_b = self.criterion(scores_b, labels_b)
        else:
            # for star shaped discriminators, to get scores per class
            labels_a = Variable(th.FloatTensor(scores_a.size(1)).fill_(self.real_label))
            labels_b = Variable(th.FloatTensor(scores_b.size(1)).fill_(self.fake_label))
            errD_a = []
            errD_b = []
            for i in range(scores_a.size(0)):
                errD_a.append(self.criterion(scores_a[i], labels_a))
                errD_b.append(self.criterion(scores_b[i], labels_b))
            errD_a = torch.cat(errD_a, 0)
            errD_b = torch.cat(errD_b, 0)

        errD = errD_a + errD_b
        return errD

    def compute_gen_score(self, data):
        th = torch.cuda if self.is_cuda else torch
        scores = self.netD(data)
        labels = Variable(th.FloatTensor(scores.size()).fill_(self.generator_label))
        errG = self.criterion(scores, labels)
        return errG