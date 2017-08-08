# GANs for Biological Image Synthesis
This codes implements the ICCV-2017 paper "GANs for Biological Image Synthesis". The paper and its supplementary materials is available on [arXiv](TODO).

This code contains the following pieces:
- implementation of [DCGAN](https://github.com/pytorch/examples/tree/master/dcgan), [WGAN](https://github.com/martinarjovsky/WassersteinGAN), [WGAN-GP](https://github.com/igul222/improved_wgan_training)
- implementation of green-on-red separable DCGAN, multi-channel DCGAN, star-shaped DCGAN (see our ICCV 2017 paper for details)
- implementation of the evaluation techniques: classifier two-samples test and reconstruction of the test set

The code is released under Apache v2 License allowing to use the code in any way you want.
For the license on the LIN dataset, please contact the authors of Dodgson et al. (2017).

As a teaser, we show our final results (animated interpolations that mimic the cell growth cycle) right away:
![lin_movie2.gif](http://www.di.ens.fr/sierra/research/biogans/lin_movie2.gif "Cell cycle reconstruction 2")
![lin_movie3.gif](http://www.di.ens.fr/sierra/research/biogans/lin_movie3.gif "Cell cycle reconstruction 3")
![lin_movie1.gif](http://www.di.ens.fr/sierra/research/biogans/lin_movie1.gif "Cell cycle reconstruction 1")

### Citation

If you are using this software please cite the following paper in any resulting publication:

Anton Osokin, Anatole Chessel, Rafael E. Carazo Salas and Federico Vaggi, GANs for Biological Image Synthesis, in proceedings of the International Conference on Computer Vision (ICCV), 2017.

>@InProceedings{osokin2017biogans,<br>
    author      = {Anton Osokin and Anatole Chessel and Rafael E. Carazo Salas and Federico Vaggi},<br>
    title       = {{GANs} for Biological Image Synthesis},<br>
    booktitle   = {Proceedings of the International Conference on Computer Vision (ICCV)},<br>
    year        = {2017} }

If you are using the LIN dataset, please, also cite this paper:

James Dodgson, Anatole Chessel, Federico Vaggi, Marco Giordan, Miki Yamamoto, Kunio Arai, Marisa Madrid, Marco Geymonat, Juan Francisco Abenza, Jose Cansado, Masamitsu Sato, Attila Csikasz-Nagy  and Rafael E. Carazo Salas, Reconstructing regulatory pathways by systematically mapping protein localization interdependency networks, bioRxiv:11674, 2017

>@article{Dodgson2017,<br>
	author = {Dodgson, James and Chessel, Anatole and Vaggi, Federico and Giordan, Marco and Yamamoto, Miki and Arai, Kunio and Madrid, Marisa and Geymonat, Marco and Abenza, Juan Francisco and Cansado, Jose and Sato, Masamitsu and Csikasz-Nagy, Attila and {Carazo Salas}, Rafael E},<br>
	title = {Reconstructing regulatory pathways by systematically mapping protein localization interdependency networks},<br>
	year = {2017},<br>
	journal = {bioRxiv:11674} }


### Authors

* [Anton Osokin](http://www.di.ens.fr/~osokin/)
* [Anatole Chessel](https://scholar.google.com/citations?user=GC8aiVsAAAAJ&hl=en)
* [Rafael E. Carazo Salas](http://research-information.bristol.ac.uk/en/persons/rafael-e-carazo-salas(a7638b29-53e4-49ba-82b5-98b21d82f41f).html)
* [Federico Vaggi](https://scholar.google.it/citations?user=rgIbvJsAAAAJ&hl=en)

### Requirements

This software was written for python v3.6.1, [pytorch](http://pytorch.org/) v0.2.0 (earlier version won't work; later versions might face some backward compatibility issues, but should work), [torchvision](https://github.com/pytorch/vision)  v0.1.8 (comes with pytorch).
Many other python packages are required, but the standard [Anaconda](https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh) installation should be sufficient.
The code was tested on Ubuntu 16.04 but should run on other systems as well.

### Usage
This code release is aimed to reproduce the results of our ICCV 2017 paper.
The experiments of this paper consist of the 4 main parts:
- training and evaluating the models on the dataset by the 6 classes merged together
- computing C2ST (classifier two-sample test) distances between real images of different classes
- training and evaluating the models that support conditioning on the class labels
- reconstructing images of the test set

By classes, we mean proteins imaged in the green channel. The 6 selected proteins include Alp14, Arp3, Cki2, Mkh1, Sid2, Tea1.

Note that rerunning all the experiements would require significant computational resources. We recommend using a cluster of GPU if you want to do that.

##### Preparations
Get the code
```
git clone https://github.com/aosokin/biogans.git
```
Mark the root folder for the code
```
cd biogans
export ROOT_BIOGANS=`pwd`
```
Download and unpack the dataset (438MB)
```
wget -P data http://www.di.ens.fr/sierra/research/biogans/LIN_Normalized_WT_size-48-80.zip
unzip data/LIN_Normalized_WT_size-48-80.zip -d data
```
If you are interested, there is a version with twice bigger images [here](http://www.di.ens.fr/sierra/research/biogans/LIN_Normalized_WT_size-96-160.zip) (1.3GB).

##### Models for 6 classes merged together
Prepare the dataset and splits for evaluation
```
cd $ROOT_BIOGANS/experiments/models_6class_joint
./make_dataset_size-48-80_6class.sh
python make_splits_size-48-80_6class.py
```
If you just want to play with the trained models, we've release the ones at iteration 500k. You can dowload the model with these lines:
```
wget -P $ROOT_BIOGANS/models/size-48-80_6class_wgangp-adam http://www.di.ens.fr/sierra/research/biogans/models/size-48-80_6class_wgangp-adam/netG_iter_500000.pth
wget -P $ROOT_BIOGANS/models/size-48-80_6class_wgangp-sep-adam http://www.di.ens.fr/sierra/research/biogans/models/size-48-80_6class_wgangp-sep-adam/netG_iter_500000.pth
wget -P $ROOT_BIOGANS/models/size-48-80_6class_gan-adam http://www.di.ens.fr/sierra/research/biogans/models/size-48-80_6class_gan-adam/netG_iter_500000.pth
wget -P $ROOT_BIOGANS/models/size-48-80_6class_gan-sep-adam http://www.di.ens.fr/sierra/research/biogans/models/size-48-80_6class_gan-sep-adam/netG_iter_500000.pth
wget -P $ROOT_BIOGANS/models/size-48-80_6class_wgan-rmsprop http://www.di.ens.fr/sierra/research/biogans/models/size-48-80_6class_wgan-rmsprop/netG_iter_500000.pth
wget -P $ROOT_BIOGANS/models/size-48-80_6class_wgan-sep-rmsprop http://www.di.ens.fr/sierra/research/biogans/models/size-48-80_6class_wgan-sep-rmsprop/netG_iter_500000.pth
```
If you want to train the models yourself (might take a while), we used these scripts to get the models reported in our paper:
```
./train_size-48-80_6class_wgangp-adam.sh
./train_size-48-80_6class_wgangp-sep-adam.sh
./train_size-48-80_6class_gan-adam.sh
./train_size-48-80_6class_gan-sep-adam.sh
./train_size-48-80_6class_wgan-rmsprop.sh
./train_size-48-80_6class_wgan-sep-rmsprop.sh
```
To perform the full C2ST evaluation presented in Figure 8, generate the job scripts
```
python make_eval_jobs_size-48-80_6class_fake_vs_real.py
python make_eval_jobs_size-48-80_6class-together_real_vs_real.py
```
and run all the scripts in `jobs_eval_6class_fake_vs_real` and `jobs_eval_6class-together_real_vs_real`. If you are interested in something specific, please, pick the jobs that you want.
After all the jobs run, one can redo our figures with `analyze_eval_6class_fake_vs_real.ipynb` and `make_figures_3and4.ipynb`.

##### C2ST for real vs. real images
Prepare the dataset and splits for evaluation
```
cd $ROOT_BIOGANS/experiments/real_vs_real
./make_dataset_size-48-80_8class.sh
python make_splits_size-48-80_6class.py
./make_splits_size-48-80_8class_real_vs_real.sh
```
Prepare all the jobs for evaluation
```
python make_eval_jobs_size-48-80_8class_real_vs_real.py
```
and runs all the scripts in `jobs_eval_8class_real_vs_real`. After this is done, you can reproduce Table 1 with `analyze_eval_8class_real_vs_real.ipynb`.

##### Models with conditioning on the class labels
Prepare the dataset and splits for evaluation
```
cd $ROOT_BIOGANS/experiments/models_6class_conditional
./make_dataset_size-48-80_6class_conditional.sh
./make_splits_size-48-80_6class_conditional.sh
```
If you just want to play with the trained models, we've release some of them at iteration 50k. You can dowload the model with these lines:
```
wget -P $ROOT_BIOGANS/models/size-48-80_6class_wgangp-star-shaped-adam http://www.di.ens.fr/sierra/research/biogans/models/size-48-80_6class_wgangp-star-shaped-adam/netG_iter_50000.pth
wget -P $ROOT_BIOGANS/models/size-48-80_6class_wgangp-independent-sep-adam http://www.di.ens.fr/sierra/research/biogans/models/size-48-80_6class_wgangp-independent-sep-adam/netG_iter_50000.pth
```
To train all the models from scratch, please, run these scripts:
```
./train_size-48-80_6class_wgangp-independent-adam.sh
./train_size-48-80_6class_wgangp-independent-sep-adam.sh
./train_size-48-80_6class_wgangp-multichannel-adam.sh
./train_size-48-80_6class_wgangp-multichannel-sep-adam.sh
./train_size-48-80_6class_wgangp-star-shaped-adam.sh
```
To train the multi-channel models, you additionally need to created the cache of nearest neighbors:
```
python $ROOT_BIOGANS/code/nearest_neighbors.py
```
Prepare evaluation scripts with
```
python make_eval_jobs_size-48-80_6class_conditional.py
```
and run all the scripts in `jobs_eval_6class_conditional_fake_vs_real`. After all of this is done, you can use `analyze_eval_6class_star-shaped_fake_vs_real.ipynb`, `make_teaser.ipynb` to reproduce Table 2 and Figure 1.
The animated vizualizations and Figure 7 are done with `cell_cycle_interpolation.ipynb`.


##### Reconstructing the test set
Prepare the dataset and splits for evaluation
```
cd $ROOT_BIOGANS/experiments/models_6class_conditional
./make_dataset_size-48-80_6class_conditional.sh
```
If you just want to play with the trained models, we've release some of them at iteration 50k. You can dowload the model with these lines:
```
wget -P $ROOT_BIOGANS/models/size-48-80_6class_gan-star-shaped-adam http://www.di.ens.fr/sierra/research/biogans/models/size-48-80_6class_gan-star-shaped-adam/netG_iter_50000.pth
wget -P $ROOT_BIOGANS/models/size-48-80_6class_wgangp-star-shaped-adam http://www.di.ens.fr/sierra/research/biogans/models/size-48-80_6class_wgangp-star-shaped-adam/netG_iter_50000.pth
wget -P $ROOT_BIOGANS/models/size-48-80_6class_gan-independent-sep-adam http://www.di.ens.fr/sierra/research/biogans/models/size-48-80_6class_gan-independent-sep-adam/netG_iter_50000.pth
wget -P $ROOT_BIOGANS/models/size-48-80_6class_wgangp-independent-sep-adam http://www.di.ens.fr/sierra/research/biogans/models/size-48-80_6class_wgangp-independent-sep-adam/netG_iter_50000.pth
```
To train all the models from scratch, please, run these scripts:
```
./train_size-48-80_6class_wgangp-star-shaped-adam.sh
./train_size-48-80_6class_wgangp-independent-sep-adam.sh
./train_size-48-80_6class_wgangp-independent-adam.sh
./train_size-48-80_6class_gan-star-shaped-adam.sh
./train_size-48-80_6class_gan-independent-sep-adam.sh
./train_size-48-80_6class_gan-independent-adam.sh
```
To run all the reconstruction experiments, please, use these scripts:
```
./reconstruction_size-48-80_6class_wgangp-star-shaped-adam.sh
./reconstruction_size-48-80_6class_wgangp-independent-sep-adam.sh
./reconstruction_size-48-80_6class_wgangp-independent-adam.sh
./reconstruction_size-48-80_6class_gan-star-shaped-adam.sh
./reconstruction_size-48-80_6class_gan-independent-sep-adam.sh
./reconstruction_size-48-80_6class_gan-independent-adam.sh
```
After all of these done, you can reproduce Table 3 and Figures 6, 10 with `analyze_reconstruction_errors.ipynb`.
