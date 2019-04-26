import os
import numpy as np
from skimage import io
from itertools import product
from collections import defaultdict
from shutil import copyfile
from argparse import ArgumentParser


def chk_mkdir(*args):
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path)


def make_mask(hypo_mask_path, nonhypo_mask_path, export_path=None):
    hypo, nonhypo = io.imread(hypo_mask_path), io.imread(nonhypo_mask_path)
    hypo[hypo == 255] = 2
    nonhypo[nonhypo == 255] = 1
    mask = np.maximum(hypo, nonhypo)
    if export_path:
        io.imsave(export_path, mask)
    else:
        return mask


def make_patches(dataset_path, export_path, patch_size=(512, 512), no_overlap=False):
    """
    Takes the data folder CONTAINING MERGED MASKS and slices the
    images and masks into patches.
    """
    # make output directories
    dataset_images_path = os.path.join(dataset_path, 'images')
    dataset_masks_path = os.path.join(dataset_path, 'masks')
    new_images_path = os.path.join(export_path, 'images')
    new_masks_path = os.path.join(export_path, 'masks')

    chk_mkdir(new_masks_path, new_images_path)

    for image_filename in os.listdir(dataset_images_path):
        # reading images
        im = io.imread(os.path.join(dataset_images_path, image_filename))
        masked_im = io.imread(os.path.join(dataset_masks_path, image_filename))
        # make new folders

        x_start = list()
        y_start = list()

        if no_overlap:
            x_step = patch_size[0]
            y_step = patch_size[1]
        else:
            x_step = patch_size[0] // 2
            y_step = patch_size[1] // 2

        for x_idx in range(0, im.shape[0] - patch_size[0] + 1, x_step):
            x_start.append(x_idx)

        if im.shape[0] - patch_size[0] - 1 > 0:
            x_start.append(im.shape[0] - patch_size[0] - 1)

        for y_idx in range(0, im.shape[1] - patch_size[1] + 1, y_step):
            y_start.append(y_idx)

        if im.shape[1] - patch_size[1] - 1 > 0:
            y_start.append(im.shape[1] - patch_size[1] - 1)

        for num, (x_idx, y_idx) in enumerate(product(x_start, y_start)):
            new_image_filename = os.path.splitext(image_filename)[0] + '_%d.png' % num
            # saving a patch of the original image
            io.imsave(
                os.path.join(new_images_path, new_image_filename),
                im[x_idx:x_idx + patch_size[0], y_idx:y_idx + patch_size[1], :]
            )
            # saving the corresponding patch of the mask
            io.imsave(
                os.path.join(new_masks_path, new_image_filename),
                masked_im[x_idx:x_idx + patch_size[0], y_idx:y_idx + patch_size[1]]
            )


def train_test_validate_split(data_path, export_path, ratios=[0.6, 0.2, 0.2]):
    dst_path = defaultdict(dict)
    for dataset, data_type in product(['train', 'test', 'validate'], ['images', 'masks']):
        set_type_path = os.path.join(export_path, dataset, data_type)
        dst_path[dataset][data_type] = set_type_path
        chk_mkdir(set_type_path)

    for image_filename in os.listdir(os.path.join(data_path, 'images')):
        src_path = {
            'images': os.path.join(data_path, 'images', image_filename),
            'masks': os.path.join(data_path, 'masks', image_filename)
        }

        dataset = np.random.choice(['train', 'test', 'validate'], p=ratios)

        for data_type in ['images', 'masks']:
            copyfile(src_path[data_type], os.path.join(dst_path[dataset][data_type], image_filename))


def imageJ_elementwise_mask_to_png(elementwise_mask_path):
    img = io.imread(elementwise_mask_path)
    folder, fname = os.path.split(elementwise_mask_path)
    fname_root, ext = os.path.splitext(fname)
    io.imsave(os.path.join(folder, fname_root + '.png'), img)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--images_folder", required=True, type=str)
    parser.add_argument("--export_folder", required=True, type=str)
    parser.add_argument("--make_patches", type=(lambda x: str(x).lower() == 'true'), default=False)

    args = parser.parse_args()

    images_folder = args.images_folder
    export_root = args.export_folder

    unet_ready_folder = os.path.join(export_root, 'converted')

    chk_mkdir(export_root, os.path.join(unet_ready_folder, 'images'),
              os.path.join(unet_ready_folder, 'masks'))

    for image_name in os.listdir(images_folder):
        hypo_mask_path = os.path.join(images_folder, image_name, '%s-hypo.png' % image_name)
        nonhypo_mask_path = os.path.join(images_folder, image_name, '%s-nonhypo.png' % image_name)
        export_path = os.path.join(unet_ready_folder, 'masks', '%s.png' % image_name)
        make_mask(hypo_mask_path, nonhypo_mask_path, export_path)
        copyfile(os.path.join(images_folder, image_name, image_name + '.png'),
                 os.path.join(unet_ready_folder, 'images', image_name + '.png'))

    if args.make_patches:
        patches_export_folder = os.path.join(export_root, 'patched_images')
        chk_mkdir(patches_export_folder)
        make_patches(unet_ready_folder, patches_export_folder, (800, 800), no_overlap=True)
