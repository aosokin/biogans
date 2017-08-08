import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

import gan, wgan, dcgan, dcgan_starshaped
import reconstruction
from custom_dataloaders import ImageFolderWithCache, CompositeImageFolder, read_all_images
from utils import RandomVerticalFlip, weights_init

# parse arguments
parser = argparse.ArgumentParser()

# main
parser.add_argument('--mode', type=str, default='train', help='mode of exacution: train | eval-gen-vs-real | eval-real-vs-real')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--random_seed', type=int, default=42,
                    help='Random seed, default - the answer to the ultimate question')

# dataset
parser.add_argument('--dataroot', type=str, required=True, help='Path to the training dataset')
parser.add_argument('--dataset_type', type=str, default='folder', help='Type of dataset: folder | folder-cached | conditional-cached | fake-multichannel')
parser.add_argument('--image_height', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--image_width', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--test_data', type=str, default='', help='Path to the test set, used for --mode evaluate')
parser.add_argument('--extra_test_sets', default=[], nargs='+', type=str, help="Extra test sets for real vs real evaluation")

#model
parser.add_argument('--model_type', type=str, required=True, help='Architecture of the model: DCGAN | DCGAN-sep | DCGAN-star-shaped | DCGAN-independent | DCGAN-sep-independent | DCGAN-multichannel | DCGAN-sep-multichannel')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--red_portion', type=float, default=0.5, help='This portion of all the channels corresponds to the red tower')
parser.add_argument('--independent_gens', action='store_true', help='Make all generators independent (fake joint model)')
parser.add_argument('--red_nn_file', type=str, default='', help='File with precomputeed nearest neighbors of the red channel (for fake-multichannel dataset)')

# training
parser.add_argument('--GAN_algorithm', type=str, default='GAN', help='GAN algorithm to train: GAN | WGAN | WGAN-GP')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--num_disc_iters', type=int, default=1, help='Number of iterations of the discriminator for one update of the generator')

# WGAN and WGAN-GP
parser.add_argument('--wgan_clamp_lower', type=float, default=-0.01, help='for WGAN')
parser.add_argument('--wgan_clamp_upper', type=float, default=0.01, help='for WGAN')
parser.add_argument('--wgangp_lambda', type=float, default=10.0, help='for WGAN-GP')

# optimization
parser.add_argument('--num_iter', type=int, default=3000, help='number of iterations to train for')
parser.add_argument('--optimizer', type=str, default='default', help='optimizer to use for training: default (depends on GAN_algorithm) | adam | rmsprop ')
parser.add_argument('--lrD', type=float, default=None, help='learning rate for Critic, default: depends on GAN_algorithm and optimizer')
parser.add_argument('--lrG', type=float, default=None, help='learning rate for Generator, default: depends on GAN_algorithm and optimizer')
parser.add_argument('--beta1', type=float, default=None, help='beta1 for adam. default: depends on GAN_algorithm and optimizer')
parser.add_argument('--beta2', type=float, default=None, help='beta2 for adam. default: depends on GAN_algorithm and optimizer')

# logging
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--save_iter', type=int, default=None, help='How often to save models')
parser.add_argument('--image_iter', type=int, default=500, help='How often to draw samples from the models')
parser.add_argument('--fixed_noise_file', type=str, default='', help='File to get shared fixed noise (to evaluate samples)')
parser.add_argument('--prefix_fake_samples', type=str, default='fake_samples', help='Fake image prefix')
parser.add_argument('--prefix_real_samples', type=str, default='real_samples', help='Fake image prefix')

# misc
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for image reading')

opt = parser.parse_args()
print(opt)

if opt.mode == 'eval-gen-vs-real':
    assert opt.netG != '', 'You need to provide trained generator to evaluate'

# create dir for experiments
if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir -p {0}'.format(opt.experiment))

# fix random seed
print("Random Seed: ", opt.random_seed)
random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)

# deal with GPUs
if opt.cuda:
    cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# make some parameters case insensitive
model_type = opt.model_type.casefold()
dataset_type = opt.dataset_type.casefold()
gan_algorithm = opt.GAN_algorithm.casefold()
optimizer_name = opt.optimizer.casefold()
execution_mode = opt.mode.casefold()

# create the dataset
opt.mean_val = 0.5
opt.std_val = 0.5

def create_dataset(data_path, red_nn_file=opt.red_nn_file):
    if dataset_type == 'folder':
        # for the regular 'folder' dataset, the normalization has to have the 3 image channels explicitly
        image_normalization = transforms.Normalize((opt.mean_val,) * 3, (opt.std_val,) * 3)
        dataset = dset.ImageFolder(root=data_path,
                                   transform=transforms.Compose([
                                       transforms.Scale((opt.image_width, opt.image_height)),
                                       transforms.CenterCrop((opt.image_height, opt.image_width)),
                                       transforms.RandomHorizontalFlip(),
                                       RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       image_normalization,
                                   ]))
    elif dataset_type == 'folder-cached':
        # for the 'folder-cached' dataset and the datasets below, the same 1-channel normalization is applied to all the channels
        image_normalization = transforms.Normalize((opt.mean_val,) * 1, (opt.std_val,) * 1)
        image_cache = read_all_images(data_path, opt.num_workers)
        dataset = ImageFolderWithCache(data_path, image_cache, do_random_flips=True,
                                       normalization=image_normalization)
    elif dataset_type == 'conditional-cached':
        image_normalization = transforms.Normalize((opt.mean_val,) * 1, (opt.std_val,) * 1)
        dataset = dcgan_starshaped.DatasetConditionalCached(data_path, do_random_flips=True,
                                                     num_workers=opt.num_workers, normalization=image_normalization)
    elif dataset_type == 'fake-multichannel':
        image_normalization = transforms.Normalize((opt.mean_val,) * 1, (opt.std_val,) * 1)
        image_cache = read_all_images(data_path, opt.num_workers)
        assert red_nn_file
        print('Reading the nn file', red_nn_file)
        nn_dict = torch.load(red_nn_file)
        dataset = CompositeImageFolder(data_path, nn_dict, image_cache, do_random_flips=True,
                                       normalization=image_normalization)
    else:
        raise RuntimeError("Unknown dataset type: {0}".format(opt.dataset_type))
    return dataset


def create_dataloader(dataset, shuffle=True):
    assert dataset
    if dataset_type == 'conditional-cached':
        dataloader = dcgan_starshaped.DataLoaderConditionalCached(dataset, batch_size=opt.batch_size, shuffle=shuffle)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=shuffle,
                                                 num_workers=opt.num_workers)
    return dataloader


# training set
dataset_train = None
if execution_mode not in ['reconstruction', 'reconstruction-separable']:
    dataset_train = create_dataset(opt.dataroot)
    dataloader_train = create_dataloader(dataset_train)
    opt.class_names = dataset_train.classes
    opt.n_classes = len(dataset_train.classes)

# test set if needed
if execution_mode in ['eval-gen-vs-real', 'eval-real-vs-real', 'reconstruction', 'reconstruction-separable']:
    assert opt.test_data
    dataset_test = create_dataset(opt.test_data)
    dataloader_test = create_dataloader(dataset_test, shuffle=False)
    if dataset_train and execution_mode in ['eval-gen-vs-real', 'reconstruction', 'reconstruction-separable']:
        # doe not check this for opt.mode == 'eval-real-vs-real', because it is done for comparing different classes
        assert(opt.class_names == dataset_test.classes)
    else:
        opt.class_names = dataset_test.classes
        opt.n_classes = len(dataset_test.classes)

# add more options
opt.original_nc = 2
opt.nc = opt.original_nc
opt.n_extra_layers = int(opt.n_extra_layers)
opt.g_input_size = opt.nz

# create the models
if gan_algorithm == 'wgan-gp':
    batch_norm_in_disc = False
else:
    batch_norm_in_disc = True

if model_type == 'dcgan':
    opt.separable_gen = False
    netG = dcgan.DCGAN_G((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ngf, opt.n_extra_layers, red_portion=None)
    netD = dcgan.DCGAN_D((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ndf, opt.n_extra_layers,
                         use_batch_norm=batch_norm_in_disc)
elif model_type == 'dcgan-sep':
    opt.separable_gen = True
    netG = dcgan.DCGAN_G((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ngf, opt.n_extra_layers, red_portion=opt.red_portion)
    netD = dcgan.DCGAN_D((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ndf, opt.n_extra_layers,
                         use_batch_norm=batch_norm_in_disc)
elif model_type in 'dcgan-star-shaped':
    opt.separable_gen = True
    netG = dcgan_starshaped.DCGAN_G_starShaped((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ngf,
                                               opt.n_classes, red_portion=opt.red_portion)
    netD = dcgan_starshaped.DCGAN_D_starShaped((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ndf,
                                               opt.n_classes, opt.n_extra_layers, use_batch_norm=batch_norm_in_disc)
elif model_type == 'dcgan-independent':
    opt.separable_gen = False
    netG = dcgan_starshaped.DCGAN_G_independent((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ngf,
                                            opt.n_classes, red_portion=None)
    netD = dcgan_starshaped.DCGAN_D_starShaped((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ndf,
                                               opt.n_classes, opt.n_extra_layers, use_batch_norm=batch_norm_in_disc)

elif model_type == 'dcgan-sep-independent':
    opt.separable_gen = True
    netG = dcgan_starshaped.DCGAN_G_independent((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ngf,
                                                opt.n_classes, red_portion=opt.red_portion)
    netD = dcgan_starshaped.DCGAN_D_starShaped((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ndf,
                                               opt.n_classes, opt.n_extra_layers, use_batch_norm=batch_norm_in_disc)
elif model_type == 'dcgan-multichannel':
    opt.nc = opt.n_classes + 1
    opt.separable_gen = False
    netG = dcgan.DCGAN_G((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ngf, opt.n_extra_layers, red_portion=None)
    netD = dcgan.DCGAN_D((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ndf, opt.n_extra_layers,
                         use_batch_norm=batch_norm_in_disc)
elif opt.model_type == 'dcgan-sep-multichannel':
    opt.nc = opt.n_classes + 1
    opt.separable_gen = True
    netG = dcgan.DCGAN_G((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ngf, opt.n_extra_layers, red_portion=opt.red_portion)
    netD = dcgan.DCGAN_D((opt.image_height, opt.image_width), opt.g_input_size, opt.nc, opt.ndf, opt.n_extra_layers,
                         use_batch_norm=batch_norm_in_disc)
else:
    raise RuntimeError("Unknown model type: {0}".format(opt.model_type))

# init the generator
netG.apply(weights_init)
if opt.netG != '':  # load checkpoint if needed
    print('Loading netG from', opt.netG)
    netG.load_state_dict(torch.load(opt.netG))

# init the discriminator
netD.apply(weights_init)
if opt.netD != '':
    print('Loading netD from', opt.netD)
    netD.load_state_dict(torch.load(opt.netD))

# adjust the models for some special cases
if model_type in ['dcgan-multichannel', 'dcgan-sep-multichannel']:
    # add-hock fix for DCGAN-multichannel evaluation
    if execution_mode == 'eval-gen-vs-real':
        #  to compare against star-shaped models we need exactly the same discriminator
        if opt.ndf == opt.ngf:
            print("""WARNING: You have opt.ndf == opt.ngf. This might a typical bug for mistake for DCGAN-multichannel and DCGAN-sep-multichannel models.
opt.ndf should depends on the evaluation metric and not on the trained model.
Please, set --ndf equal to the one of the star-shaped your are evaluating against.""")
        opt.nc = opt.original_nc
        netG = dcgan_starshaped.DCGAN_G_starShaped_from_multi_channel(opt.n_classes, netG)
        netD = dcgan_starshaped.DCGAN_D_starShaped((opt.image_height, opt.image_width), None, opt.original_nc,
                                                   opt.ndf, opt.n_classes,
                                                   opt.n_extra_layers, use_batch_norm=batch_norm_in_disc)

    if execution_mode in ['reconstruction', 'reconstruction-separable']:
        # add a wrapper on top of the generator to have the model in the right format
        opt.nc = opt.original_nc
        netG = dcgan_starshaped.DCGAN_G_starShaped_from_multi_channel(opt.n_classes, netG)

# print the models to examine them
print(netG)
print(netD)


def gan_choice(opt_dict, param_name='parameter'):
    if gan_algorithm in opt_dict:
        pick = opt_dict[gan_algorithm]
    else:
        raise RuntimeError("Unknown value of {1}: {0}".format(opt.GAN_algorithm, param_name))
    return pick

# setup optimizer
if optimizer_name == 'default':
    optimizer_name = gan_choice({ 'gan': 'adam', 'wgan': 'rmsprop', 'wgan-gp': 'adam'}, 'optimizer')

if opt.lrD is None:
    opt.lrD = gan_choice({ 'gan': 0.0002, 'wgan': 0.00005, 'wgan-gp': 0.0001}, 'lrD')

if opt.lrG is None:
    opt.lrG = gan_choice({ 'gan': 0.0002, 'wgan': 0.00005, 'wgan-gp': 0.0001}, 'lrG')

if opt.beta1 is None:
    opt.beta1 = gan_choice({ 'gan': 0.5, 'wgan': 0.0, 'wgan-gp': 0.0}, 'beta1')

if opt.beta2 is None:
    opt.beta2 = gan_choice({ 'gan': 0.999, 'wgan': 0.9, 'wgan-gp': 0.9}, 'beta2')

if optimizer_name == 'adam':
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
elif optimizer_name == 'rmsprop':
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)
else:
    raise (RuntimeError("Do not recognize optimizer %s" % opt.optimizer))

# create the GAN class
gan_model = gan_choice({'gan': gan.GAN(netG, netD, optimizerD, optimizerG, opt),
                        'wgan': wgan.WGAN(netG, netD, optimizerD, optimizerG, opt),
                        'wgan-gp': wgan.WGANGP(netG, netD, optimizerD, optimizerG, opt)},
                       'GAN_algorithm')

# the main operation
if execution_mode == 'train':
    # train the model
    gan_model.train(dataloader_train, opt)
elif execution_mode == 'eval-gen-vs-real':
    # save samples
    fixed_noise = gan_model.generate_fixed_noise(opt)
    gan_model.save_samples(None, fixed_noise, None, opt)
    # train an independent discriminator
    gan_model.train_disc_fake_vs_data(dataloader_train, opt)
    # compute the two-sample score
    score = -gan_model.compute_dics_score_fake_vs_data(dataloader_test, opt)
    if score.numel() > 1:
        assert(score.numel() == len(opt.class_names))
        print('{0}-scores per class.'.format(opt.GAN_algorithm), end=' ')
        for i_cl, cl in enumerate(opt.class_names):
            print('{0}: {1};'.format(cl, score[i_cl]), end=' ')
        print('')
    score = score.mean()
    print('{0}-score: equals {1}'.format(opt.GAN_algorithm, score))
elif execution_mode == 'eval-real-vs-real':
    assert(len(opt.extra_test_sets) == 2)
    assert(opt.dataset_type == 'folder-cached' or opt.dataset_type == 'folder')
    # train an independent discriminator
    gan_model.train_disc_data_vs_data(dataloader_train, dataloader_test, opt)
    # prepare extra test sets
    dataloaders_extra = []
    for i in range(len(opt.extra_test_sets)):
        dataset_extra = create_dataset(opt.extra_test_sets[i])
        dataloaders_extra.append(create_dataloader(dataset_extra))
    # compute the two-sample score
    score = -gan_model.compute_dics_score_data_vs_data(dataloaders_extra[0], dataloaders_extra[1], opt)
    score = score.mean()
    print('{0}-score: equals {1}'.format(opt.GAN_algorithm, score))
elif execution_mode == 'reconstruction':
    reconstruction.run_experiment(gan_model.netG, dataloader_test, opt.dataroot, opt, optimize_red_first=False)
elif execution_mode == 'reconstruction-separable':
    reconstruction.run_experiment(gan_model.netG, dataloader_test, opt.dataroot, opt, optimize_red_first=True)
else:
    raise RuntimeError("Unknown mode: {0}".format(opt.mode))
