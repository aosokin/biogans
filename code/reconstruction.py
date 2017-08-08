import os, sys
from tqdm import tqdm
import numpy as np
from scipy.stats import multivariate_normal

import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils

from gan import GanImageloader
from utils import pad_channels
import nearest_neighbors


def run_experiment(netG, dataloader_test, nn_path, opt, optimize_red_first=False, n_bfgs_iter=50, lbfgs_lr=0.05, num_lbfgs_trials=5):
    """
    Optimize over the latent noise to try to reconstruct a target image.
    """
    # read the training set to get images for nearest neighbors
    search_images, search_image_names = nearest_neighbors.read_search_images(search_dataset=nn_path, classes_of_interest=opt.class_names,
                                                                             num_channels=opt.original_nc, use_cuda=opt.cuda)
    # prepare data loader
    data_looper = GanImageloader()
    iterator_data = data_looper.return_iterator(dataloader_test, opt.cuda, opt.nc, return_labels=False, num_passes=1)

    # the generator in the eval mode
    netG.eval()

    # create folders
    for cl in opt.class_names:
        os.system('mkdir -p {0}'.format(os.path.join(opt.experiment, cl)))

    l2_dists = {cl: [] for cl in opt.class_names}
    lls_noise = {cl: [] for cl in opt.class_names}
    lls_noise_init = {cl: [] for cl in opt.class_names}
    nn_dists = {cl: [] for cl in opt.class_names}
    mvn = multivariate_normal(np.zeros(opt.nz), np.identity(opt.nz))


    def compute_rec_error(data, rec):
        rec_error = data - rec
        l2_dist = torch.sum(rec_error ** 2 / rec_error.numel()) ** 0.5
        return l2_dist

    def im_to_01(im):
        return im * opt.std_val + opt.mean_val

    def im_to_original(im):
        return (im - opt.mean_val) / opt.std_val


    for i_batch, data in enumerate(iterator_data):
        print('Batch {}'.format(i_batch))
        assert (len(opt.class_names) == data.size(0))
        assert (data.dim() == 5)
        for i_class in range(data.size(0)):
            class_name = opt.class_names[i_class]

            # add class information to the generator
            netG_forward = lambda input: netG(input, i_class)

            reconstructions_best = data[i_class].clone()
            reconstructions_best.data.zero_()
            reconstructions_best_init = data[i_class].clone()
            reconstructions_best_init.data.zero_()
            reconstructions_error_best = [float('inf')] * data[i_class].size(0)
            ll_noise_best = [float('inf')] * data[i_class].size(0)
            ll_noise_init_best = [float('inf')] * data[i_class].size(0)
            nn_dists_batch = [float('inf')] * data[i_class].size(0)


            for i_trial in range(num_lbfgs_trials):
                print('Class {0}: {1}, trial {2} of {3}'.format(i_class, class_name, i_trial + 1, num_lbfgs_trials))
                sys.stdout.flush()
                # get the noise leading to the good reconstruction
                if optimize_red_first:
                    noise_init, noise = reconstruct_cells_red_first(data[i_class], netG_forward, opt, n_bfgs_iter=n_bfgs_iter, lbfgs_lr=lbfgs_lr)
                else:
                    noise_init, noise = reconstruct_cells(data[i_class], netG_forward, opt, n_bfgs_iter=n_bfgs_iter, lbfgs_lr=lbfgs_lr)

                # get reconstructions
                reconstructions_init = netG_forward(noise_init)
                reconstructions = netG_forward(noise)

                # compute reconstruction errors
                for i_im in range(reconstructions.size(0)):
                    # get log-likelihoods
                    noise_np = noise[i_im].view(-1).data.cpu().numpy()
                    ll_noise = -mvn.logpdf(noise_np)
                    noise_init_np = noise_init[i_im].view(-1).data.cpu().numpy()
                    ll_noise_init = -mvn.logpdf(noise_init_np)


                    l2_dist = compute_rec_error(im_to_01(data[i_class][i_im].data), im_to_01(reconstructions[i_im].data))

                    if l2_dist < reconstructions_error_best[i_im]:
                        reconstructions_error_best[i_im] = l2_dist
                        reconstructions_best[i_im] = reconstructions[i_im]
                        reconstructions_best_init[i_im] = reconstructions_init[i_im]
                        ll_noise_best[i_im] = ll_noise
                        ll_noise_init_best[i_im] = ll_noise_init

            # find nearest neighbors from the training set
            neighbors = torch.FloatTensor(reconstructions_best.size())
            if opt.cuda:
                neighbors = neighbors.cuda()
            for i_im in range(reconstructions_best.size(0)):
                ref_im = data[i_class][i_im].data
                ref_im_01 = im_to_01(ref_im)
                _, nn_ids = nearest_neighbors.find_neighbors(ref_im_01, search_images[class_name], search_image_names[class_name], num_neighbors=1)
                nn_im_01 = search_images[class_name][nn_ids[0]]
                neighbors[i_im] = im_to_original(nn_im_01)
                nn_dists_batch[i_im] = compute_rec_error(nn_im_01, ref_im_01)
            neighbors = Variable(neighbors)

            # save results
            for i_im in range(reconstructions_best.size(0)):
                all_images = [data[i_class][i_im], reconstructions_best[i_im], reconstructions_best_init[i_im], neighbors[i_im]]
                all_images = torch.stack(all_images, 0)
                all_images = pad_channels(all_images.data, 3)
                file_name = os.path.join(opt.experiment, class_name, '{0}_batch{1}_image{2}.png'.format(class_name, i_batch, i_im))
                vutils.save_image(im_to_01(all_images), file_name)

                l2_dist = reconstructions_error_best[i_im]
                ll_noise = ll_noise_best[i_im]
                ll_noise_init = ll_noise_init_best[i_im]

                l2_dists[class_name].append(l2_dist)
                lls_noise[class_name].append(ll_noise)
                lls_noise_init[class_name].append(ll_noise_init)
                nn_dists[class_name].append(nn_dists_batch[i_im])

    # saving the full reconstruction data
    all_data = {'l2_dists': l2_dists, 'lls_noise': lls_noise, 'lls_noise_init': lls_noise_init, 'nn_dists': nn_dists}
    torch.save(all_data, os.path.join(opt.experiment,  'reconstruction_data.pth'))

    # print aggregated statistics
    for i_class, class_name in enumerate(opt.class_names):
        l2 = np.array(l2_dists[class_name])
        l2_mean = np.mean(l2)
        l2_std = np.std(l2)
        ll_noise = np.array(lls_noise[class_name])
        ll_noise_mean = np.mean(ll_noise)
        ll_noise_std = np.std(ll_noise)
        ll_noise_init = np.array(lls_noise_init[class_name])
        ll_noise_init_mean = np.mean(ll_noise_init)
        ll_noise_init_std = np.std(ll_noise_init)
        nn_d = np.array(nn_dists[class_name])
        nn_d_mean = np.mean(nn_d)
        nn_d_std = np.std(nn_d)
        print('Class {0}: L2-reconstr mean {1:0.3f} std {2:0.3f}; L2-noise mean {3:0.3f} std {4:0.3f}; L2-noise-init mean {5:0.3f} std {6:0.3f}; NN dist {7:0.3f} std {8:0.3f}'.format(class_name, l2_mean, l2_std, ll_noise_mean, ll_noise_std, ll_noise_init_mean, ll_noise_init_std, nn_d_mean, nn_d_std))

    l2 = np.concatenate([np.array(d) for d in l2_dists.values()])
    l2_mean = np.mean(l2)
    l2_std = np.std(l2)
    ll_noise = np.concatenate([np.array(d) for d in lls_noise.values()])
    ll_noise_mean = np.mean(ll_noise)
    ll_noise_std = np.std(ll_noise)
    ll_noise_init = np.concatenate([np.array(d) for d in lls_noise_init.values()])
    ll_noise_init_mean = np.mean(ll_noise_init)
    ll_noise_init_std = np.std(ll_noise_init)
    nn_d = np.concatenate([np.array(d) for d in nn_dists.values()])
    nn_d_mean = np.mean(nn_d)
    nn_d_std = np.std(nn_d)

    print('All classes: L2-reconstr mean {0:0.3f} std {1:0.3f}; L2-noise mean {2:0.3f} std {3:0.3f}; L2-noise-init mean {4:0.3f} std {5:0.3f}; NN dist {6:0.3f} std {7:0.3f}'.format(l2_mean, l2_std, ll_noise_mean, ll_noise_std, ll_noise_init_mean, ll_noise_init_std, nn_d_mean, nn_d_std))


def reconstruct_cells(imgs, netG, opt, n_bfgs_iter=100, lbfgs_lr=0.1):
    noise = torch.FloatTensor(int(opt.batch_size), opt.nz, 1, 1)
    noise.normal_(0, 1)

    if opt.cuda:
        noise = noise.cuda()

    noise = Variable(noise)
    noise.requires_grad = True
    noise_init = noise.clone()

    optim_input = optim.LBFGS([noise], lr=lbfgs_lr)

    def closure():
        optim_input.zero_grad()
        gen_img = netG(noise)

        l2_loss = torch.mean((imgs - gen_img) ** 2)
        l2_loss.backward()
        # print(l2_loss.data[0])
        # sys.stdout.flush()
        return l2_loss

    # Do the optimization across batch
    for i in tqdm(range(n_bfgs_iter)):
        optim_input.step(closure)
    return noise_init, noise


def reconstruct_cells_red_first(imgs, netG, opt, n_bfgs_iter=100, lbfgs_lr=0.1):
    assert(imgs.size(1) == 2)
    n_red_noise = int(opt.nz * opt.red_portion)
    n_green_noise = opt.nz - n_red_noise
    red_noise = torch.FloatTensor(int(opt.batch_size), n_red_noise, 1, 1).normal_(0, 1)
    green_noise = torch.FloatTensor(int(opt.batch_size), n_green_noise, 1, 1).normal_(0, 1)

    if opt.cuda:
        red_noise = red_noise.cuda()
        green_noise = green_noise.cuda()

    red_noise = Variable(red_noise)
    red_noise.requires_grad = True
    green_noise = Variable(green_noise)
    green_noise.requires_grad = False

    noise = torch.cat([red_noise, green_noise], 1)
    noise_init = noise.clone()

    optim_input = optim.LBFGS([red_noise], lr=lbfgs_lr)

    def red_closure():
        optim_input.zero_grad()
        noise = torch.cat([red_noise, green_noise], 1)
        gen_img = netG(noise)

        l2_loss = torch.mean((imgs[:, 0, :, :] - gen_img[:, 0, :, :]) ** 2)
        l2_loss.backward()
        # print(l2_loss.data[0])
        # sys.stdout.flush()
        return l2_loss

    # Do the optimization across batch
    for i in tqdm(range(n_bfgs_iter)):
        optim_input.step(red_closure)

    # Optimization across green channel now
    red_noise.requires_grad = False
    green_noise.requires_grad = True
    optim_input = optim.LBFGS([green_noise], lr=lbfgs_lr)

    def green_closure():
        optim_input.zero_grad()
        noise = torch.cat([red_noise, green_noise], 1)
        gen_img = netG(noise)

        l2_loss = torch.mean((imgs[:, 1, :, :] - gen_img[:, 1, :, :]) ** 2)
        l2_loss.backward()
        # print(l2_loss.data[0])
        # sys.stdout.flush()
        return l2_loss

    for i in tqdm(range(n_bfgs_iter)):
        optim_input.step(green_closure)

    # do not forget this because real variables are red_noise and green_noise
    noise = torch.cat([red_noise, green_noise], 1)

    return noise_init, noise
