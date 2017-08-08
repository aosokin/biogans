import os
import random

classes_of_interest = ['Alp14', 'Arp3', 'Cki2', 'Mkh1', 'Sid2', 'Tea1']

data_root = '../../data'
data_paths = ['LIN_Normalized_WT_size-48-80_6class_train',
              'LIN_Normalized_WT_size-48-80_6class_test']
target_folders = ['LIN_Normalized_WT_size-48-80_6class_train_splits',
                  'LIN_Normalized_WT_size-48-80_6class_test_splits']

for data_path, target_folder in zip(data_paths, target_folders):
    data_path = os.path.join(data_root, data_path)
    target_folder = os.path.join(data_root, target_folder)
    print('Working with', data_path)

    os.mkdir(target_folder)

    # split number, split role (train or test), class_id
    split_path = os.path.join(target_folder, 'split{0}', '{1}', '{2}')

    random_seed = 1
    num_splits = 10
    num_classes = len(classes_of_interest)

    for class_id in classes_of_interest:
        for i_split in range(num_splits):
            print('Class {0}, split {1}'.format(class_id, i_split))

            random.seed(i_split + random_seed)

            class_path = os.path.join(data_path, class_id)
            class_images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            class_images.sort()

            random.shuffle(class_images)

            test_path = split_path.format(i_split, 'test', class_id)
            train_path = split_path.format(i_split, 'train', class_id)
            os.makedirs(test_path)
            os.makedirs(train_path)

            num_test = len(class_images) // 2
            for i_im in range(num_test):
                os.system('ln -rs {0} {1}'.format(os.path.join(class_path, class_images[i_im]),
                                                  os.path.join(test_path, class_images[i_im])))

            for i_im in range(num_test, len(class_images)):
                os.system('ln -rs {0} {1}'.format(os.path.join(class_path, class_images[i_im]),
                                                  os.path.join(train_path, class_images[i_im])))
