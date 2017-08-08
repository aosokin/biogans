import torch
from torch.autograd import Variable
import torch.utils.data

from gan import GAN_base

class WGAN(GAN_base):
    def __init__(self, netG, netD, optimizerD, optimizerG, opt):
        GAN_base.__init__(self, netG, netD, optimizerD, optimizerG, opt)
        self.clamp_lower = opt.wgan_clamp_lower
        self.clamp_upper = opt.wgan_clamp_upper

    def clamp_critic_parameters(self):
        # clamp parameters to a cube
        for p in self.netD.parameters():
            p.data.clamp_(self.clamp_lower, self.clamp_upper)

    def compute_disc_score(self, data_a, data_b):
        scores_a = self.netD(data_a)
        scores_b = self.netD(data_b)

        mean_dim = 0 if  scores_a.dim() == 1 else 1
        errD = scores_a.mean(mean_dim) - scores_b.mean(mean_dim)
        return errD

    def compute_gen_score(self, data):
        return self.netD(data).mean()

    def train_D_one_step(self, iterator_data, iterator_fake):
        errD, data_a, data_b = GAN_base.train_D_one_step(self, iterator_data, iterator_fake)
        # clamp parameters
        self.clamp_critic_parameters()
        return errD, data_a, data_b

    def train_one_step(self, iterator_data, iterator_fake, num_disc_iters=1, i_iter=None):
        # train the discriminator num_disc_iters times
        if i_iter < 25 or i_iter % 500 == 0:
            num_disc_iters = 100
        return GAN_base.train_one_step(self, iterator_data, iterator_fake, num_disc_iters=num_disc_iters)


class WGANGP(GAN_base):
    def __init__(self, netG, netD, optimizerD, optimizerG, opt):
        GAN_base.__init__(self, netG, netD, optimizerD, optimizerG, opt)
        self.wgangp_lambda = opt.wgangp_lambda

    def compute_gradient_penalties(self, netD, real_data, fake_data):
        # this code is base on https://github.com/caogang/wgan-gp

        # equalize batch sizes
        if real_data.dim() == 4:
            batch_size = min(real_data.size(0), fake_data.size(0))
            real_data = real_data[:batch_size]
            fake_data = fake_data[:batch_size]
            # get noisy inputs
            eps = torch.rand(batch_size)
            while eps.dim() < real_data.dim():
                eps = eps.unsqueeze(-1)
        elif real_data.dim() == 5:
            assert(real_data.size(0) == fake_data.size(0))
            batch_size = min(real_data.size(1), fake_data.size(1))
            real_data = real_data[:, :batch_size, :, :, :]
            fake_data = fake_data[:, :batch_size, :, :, :]
            # get noisy inputs
            eps = torch.rand(real_data.size(0), batch_size)
            while eps.dim() < real_data.dim():
                eps = eps.unsqueeze(-1)
        else:
            raise RuntimeError("Unknown dimension of image data: {0}".format(real_data.dim()))


        eps = eps.cuda() if real_data.is_cuda else eps
        interpolates = eps * real_data + (1 - eps) * fake_data
        interpolates = Variable(interpolates, requires_grad=True)

        # push thorugh network
        D_interpolates = netD(interpolates)

        # compute the gradients
        grads = torch.ones(D_interpolates.size())
        grads = grads.cuda() if real_data.is_cuda else grads
        gradients = torch.autograd.grad(outputs=D_interpolates, inputs=interpolates, grad_outputs=grads,
                                        create_graph=True, only_inputs=True)
        if real_data.dim() == 4:
            gradient_input = gradients[0].view(batch_size, -1)
        else:
            gradient_input = gradients[0].view(real_data.size(0) * batch_size, -1)

        # compute the penalties
        gradient_penalties = (gradient_input.norm(2, dim=1) - 1) ** 2

        if real_data.dim() != 4:
            gradient_penalties =  gradient_penalties.view(real_data.size(0), batch_size)

        return gradient_penalties

    def compute_disc_score(self, data_a, data_b):
        scores_a = self.netD(data_a)
        scores_b = self.netD(data_b)
        gradient_penalties = self.compute_gradient_penalties(self.netD, data_a.data, data_b.data)

        mean_dim = 0 if scores_a.dim() == 1 else 1
        gradient_penalty = gradient_penalties.mean(mean_dim)
        errD = scores_a.mean(mean_dim) - scores_b.mean(mean_dim) + self.wgangp_lambda * gradient_penalty

        return errD

    def compute_gen_score(self, data):
        return self.netD(data).mean()
