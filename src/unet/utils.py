import os
import numpy as np
import pandas as pd
import torch

from skimage import io, img_as_uint
from skimage.morphology import skeletonize_3d

from numbers import Number
from itertools import product

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss
from torchvision import transforms as T
from torchvision.transforms import functional as F

from .rprops import get_hypo_rprops, visualize_regions


def chk_mkdir(*args):
    for path in args:
        if path is not None and not os.path.exists(path):
            os.makedirs(path)


def dpi_to_dpm(dpi):
    # small hack, default value for dpi is False
    if not dpi:
        return False

    return dpi/25.4


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def joint_to_long_tensor(image, mask):
    return to_long_tensor(image), to_long_tensor(mask)


def make_transform(
        crop=(256, 256), p_flip=0.5, p_color=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1),
        p_random_affine=0.0, rotate_range=False, normalize=False, long_mask=False
):

    if color_jitter_params is not None:
        color_tf = T.ColorJitter(*color_jitter_params)
    else:
        color_tf = None

    if normalize:
        tf_normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(1, 1, 1))

    def joint_transform(image, mask):
        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # random crop
        if crop:
            i, j, h, w = T.RandomCrop.get_params(image, crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
            if np.random.rand() < p_flip:
                image, mask = F.hflip(image), F.hflip(mask)

        # color transforms || ONLY ON IMAGE
        if color_tf is not None:
            if np.random.rand() < p_color:
                image = color_tf(image)

        # random rotation
        if rotate_range and not p_random_affine:
            if np.random.rand() < 0.5:
                angle = rotate_range * (np.random.rand() - 0.5)
                image, mask = F.rotate(image, angle), F.rotate(mask, angle)

        # random affine
        if np.random.rand() < p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)

        # transforming to tensor
        image = F.to_tensor(image)
        if not long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        # normalizing image
        if normalize:
            image = tf_normalize(image)

        return image, mask

    return joint_transform


def confusion_matrix(prediction, target, n_classes):
    """
    prediction, target: torch.Tensor objects
    """
    prediction = torch.argmax(prediction, dim=0).long()
    target = torch.squeeze(target, dim=0)

    conf_mtx = torch.zeros(n_classes, n_classes).long()
    for i, j in product(range(n_classes), range(n_classes)):
        conf_mtx[i, j] = torch.sum((prediction == j) * (target == i))

    return conf_mtx


class SoftDiceLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        if weight is None:
            weight = torch.tensor(1)
        else:
            # creating tensor if needed
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight)
            # normalizing weights
            weight /= torch.sum(weight)

        super(SoftDiceLoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, y_pred, y_gt):
        """
        Args:
            y_pred: torch.Tensor of shape (n_batch, n_classes, image.shape)
            y_gt: torch.LongTensor of shape (n_batch, image.shape)
        """
        dims = (0, *range(2, len(y_pred.shape)))

        y_gt = torch.zeros_like(y_pred).scatter_(1, y_gt[:, None, :], 1)
        numerator = 2 * torch.sum(y_pred * y_gt, dim=dims)
        denominator = torch.sum(y_pred * y_pred + y_gt * y_gt, dim=dims)
        return torch.sum((1 - numerator / denominator)*self.weight)


class LogNLLLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean',
                 ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        input = torch.log(input)
        return cross_entropy(input, target, weight=self.weight, reduction=self.reduction,
                             ignore_index=self.ignore_index)


class ReadTrainDataset(Dataset):
    """
    Structure of the dataset should be:

    dataset_path
      |--images
          |--img001.png
          |--img002.png
      |--masks
          |--img001.png
          |--img002.png

    """

    def __init__(self, dataset_path, transform=None, one_hot_mask=False, long_mask=True):
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, 'images')
        self.masks_path = os.path.join(dataset_path, 'masks')
        self.images_list = os.listdir(self.images_path)

        self.transform = transform
        self.one_hot_mask = one_hot_mask
        self.long_mask = long_mask

    def __len__(self):
        return len(os.listdir(self.images_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        image = io.imread(os.path.join(self.images_path, image_filename))
        mask = io.imread(os.path.join(self.masks_path, image_filename))
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image = F.to_tensor(image)
            if self.long_mask:
                mask = to_long_tensor(F.to_pil_image(mask))
            else:
                mask = F.to_tensor(mask)

        if self.one_hot_mask:
            assert self.one_hot_mask >= 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        return image, mask, image_filename


class ReadTestDataset(Dataset):
    """
    Structure of the dataset should be:

    dataset_path
      |--images
          |--img001.png
          |--img002.png

    """

    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, 'images')
        self.images_list = os.listdir(self.images_path)

        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.images_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]
        image = io.imread(os.path.join(self.images_path, image_filename))

        if self.transform:
            image = self.transform(image)
        else:
            image = F.to_tensor(image)

        return image, image_filename


class ModelWrapper:
    def __init__(
            self, model, results_folder, loss=None, optimizer=None,
            scheduler=None, cuda_device=None
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.results_folder = results_folder
        chk_mkdir(self.results_folder)

        self.cuda_device = cuda_device
        if self.cuda_device:
            self.model.to(device=self.cuda_device)
            try:
                self.loss.to(device=self.cuda_device)
            except AttributeError:
                pass

    def train_model(self, dataset, n_epochs, n_batch=1, verbose=False,
                    validation_dataset=None, prediction_dataset=None,
                    save_freq=100):
        self.model.train(True)

        # logging losses
        loss_df = pd.DataFrame(np.zeros(shape=(n_epochs, 2)), columns=['train', 'validate'], index=range(n_epochs))

        min_loss = np.inf
        total_running_loss = 0
        for epoch_idx in range(n_epochs):

            epoch_running_loss = 0
            for batch_idx, (X_batch, y_batch, name) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=True)):
                if self.cuda_device:
                    X_batch = Variable(X_batch.to(device=self.cuda_device))
                    y_batch = Variable(y_batch.to(device=self.cuda_device))
                else:
                    X_batch, y_batch = Variable(X_batch), Variable(y_batch)

                # training
                self.optimizer.zero_grad()
                y_out = self.model(X_batch)
                training_loss = self.loss(y_out, y_batch)
                training_loss.backward()
                self.optimizer.step()

                epoch_running_loss += training_loss.item()

                if verbose:
                    print('(Epoch no. %d, batch no. %d) loss: %f' % (epoch_idx, batch_idx, training_loss.item()))

            total_running_loss += epoch_running_loss/(batch_idx + 1)
            print('(Epoch no. %d) loss: %f' % (epoch_idx, epoch_running_loss/(batch_idx + 1)))
            loss_df.loc[epoch_idx, 'train'] = epoch_running_loss/(batch_idx + 1)

            if validation_dataset is not None:
                validation_error = self.validate(validation_dataset, n_batch=1)
                loss_df.loc[epoch_idx, 'validate'] = validation_error
                if validation_error < min_loss:
                    torch.save(self.model.state_dict(), os.path.join(self.results_folder, 'model'))
                    print('Validation loss improved from %f to %f, model saved to %s'
                          % (min_loss, validation_error, self.results_folder))
                    min_loss = validation_error

                if self.scheduler is not None:
                    self.scheduler.step(validation_error)

            else:
                if epoch_running_loss/(batch_idx + 1) < min_loss:
                    torch.save(self.model.state_dict(), os.path.join(self.results_folder, 'model'))
                    print('Training loss improved from %f to %f, model saved to %s'
                          % (min_loss, epoch_running_loss / (batch_idx + 1), self.results_folder))
                    min_loss = epoch_running_loss / (batch_idx + 1)

                if self.scheduler is not None:
                    self.scheduler.step(epoch_running_loss / (batch_idx + 1))

            # saving model and logs
            loss_df.to_csv(os.path.join(self.results_folder, 'loss.csv'))
            if epoch_idx % save_freq == 0:
                epoch_save_path = os.path.join(self.results_folder, '%d' % epoch_idx)
                chk_mkdir(epoch_save_path)
                torch.save(self.model.state_dict(), os.path.join(epoch_save_path, 'model'))
                if prediction_dataset:
                    self.predict_large_images(prediction_dataset, epoch_save_path)

        self.model.train(False)

        del X_batch, y_batch

        return total_running_loss/n_batch

    def validate(self, dataset, n_batch=1):
        self.model.train(False)

        total_running_loss = 0
        for batch_idx, (X_batch, y_batch, name) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=False)):

            if self.cuda_device:
                X_batch = Variable(X_batch.to(device=self.cuda_device))
                y_batch = Variable(y_batch.to(device=self.cuda_device))
            else:
                X_batch, y_batch = Variable(X_batch), Variable(y_batch)

            y_out = self.model(X_batch)
            training_loss = self.loss(y_out, y_batch)

            total_running_loss += training_loss.item()

        print('Validation loss: %f' % (total_running_loss / (batch_idx + 1)))
        self.model.train(True)

        del X_batch, y_batch

        return total_running_loss/(batch_idx + 1)

    def predict(self, dataset, export_path, channel=None):
        self.model.train(False)
        chk_mkdir(export_path)

        for batch_idx, (X_batch, image_filename) in enumerate(DataLoader(dataset, batch_size=1)):
            if self.cuda_device:
                X_batch = Variable(X_batch.to(device=self.cuda_device))
                y_out = self.model(X_batch).cpu().data.numpy()
            else:
                X_batch = Variable(X_batch)
                y_out = self.model(X_batch).data.numpy()

            if channel:
                try:
                    io.imsave(os.path.join(export_path, image_filename[0]), y_out[0, channel, :, :])
                except:
                    print('something went wrong upon prediction')

            else:
                try:
                    io.imsave(os.path.join(export_path, image_filename[0]), y_out[0, :, :, :].transpose((1, 2, 0)))
                except:
                    print('something went wrong upon prediction')

    def predict_large_images(self, dataset, export_path=None, channel=None, tile_res=(512, 512)):
        self.model.train(False)
        if export_path:
            chk_mkdir(export_path)
        else:
            results = []

        for batch_idx, (X_batch, image_filename) in enumerate(DataLoader(dataset, batch_size=1)):
            out = self.predict_single_large_image(X_batch, channel=channel, tile_res=tile_res)

            if export_path:
                io.imsave(os.path.join(export_path, image_filename[0]), out)
            else:
                results.append(out)

        if not export_path:
            return results

    def predict_single_large_image(self, X_image, channel=None, tile_res=(512, 512)):
        image_res = X_image.shape
        # placeholder for output
        y_out_full = np.zeros(shape=(1, 3, image_res[2], image_res[3]))
        # generate tile coordinates
        tile_x = list(range(0, image_res[2], tile_res[0]))[:-1] + [image_res[2] - tile_res[0]]
        tile_y = list(range(0, image_res[3], tile_res[1]))[:-1] + [image_res[3] - tile_res[1]]
        tile = product(tile_x, tile_y)
        # predictions
        for slice in tile:

            if self.cuda_device:
                X_in = X_image[:, :, slice[0]:slice[0] + tile_res[0], slice[1]:slice[1] + tile_res[1]].to(
                    device=self.cuda_device)
                X_in = Variable(X_in)
            else:
                X_in = X_image[:, :, slice[0]:slice[0] + tile_res[0], slice[1]:slice[1] + tile_res[1]]
                X_in = Variable(X_in)

            y_out = self.model(X_in).cpu().data.numpy()
            y_out_full[0, :, slice[0]:slice[0] + tile_res[0], slice[1]:slice[1] + tile_res[1]] = y_out

        # save image
        if channel:
            out = y_out_full[0, channel, :, :]
        else:
            out = y_out_full[0, :, :, :].transpose((1, 2, 0))

        return out

    def measure_large_images(self, dataset, visualize_bboxes=False, filter=True, export_path=None,
                             skeleton_method=skeletonize_3d, dpm=False, verbose=False, tile_res=(512, 512)):
        hypocotyl_lengths = dict()
        chk_mkdir(export_path)

        assert any(isinstance(dpm, tp) for tp in [str, bool, Number]), 'dpm must be string, bool or Number'

        for batch_idx, (X_batch, image_filename) in enumerate(DataLoader(dataset, batch_size=1)):

            if verbose:
                print("Measuring %s" % image_filename[0])

            hypo_segmented = self.predict_single_large_image(X_batch, tile_res=tile_res)
            hypo_segmented_mask = hypo_segmented[:, :, 2]
            hypo_result, hypo_skeleton = get_hypo_rprops(hypo_segmented_mask, filter=filter, return_skeleton=True,
                                                         skeleton_method=skeleton_method,
                                                         dpm=dpm)
            hypo_df = hypo_result.make_df()

            hypocotyl_lengths[image_filename] = hypo_df

            if export_path:
                if visualize_bboxes:
                    hypo_img = X_batch[0].cpu().data.numpy().transpose((1, 2, 0))
                    # original image
                    visualize_regions(hypo_img, hypo_result,
                                      os.path.join(export_path, image_filename[0][:-4] + '.png'))
                    # segmentation
                    visualize_regions(hypo_segmented, hypo_result,
                                      os.path.join(export_path, image_filename[0][:-4] + '_segmentation.png'),
                                      bbox_color='0.5')
                    # skeletonization
                    visualize_regions(hypo_skeleton, hypo_result,
                                      os.path.join(export_path, image_filename[0][:-4] + '_skeleton.png'))

                hypocotyl_lengths[image_filename].to_csv(os.path.join(export_path, image_filename[0][:-4] + '.csv'),
                                                         header=True, index=True)

        return hypocotyl_lengths

    def score_large_images(self, dataset, export_path, visualize_bboxes=False, visualize_histograms=False,
                           visualize_segmentation=False,
                           filter=True, skeletonized_gt=False, match_threshold=0.5, tile_res=(512, 512),
                           dpm=False):
        chk_mkdir(export_path)

        scores = {}

        assert any(isinstance(dpm, tp) for tp in [str, bool, Number]), 'dpm must be string, bool or Number'

        if isinstance(dpm, str):
            dpm_df = pd.read_csv(dpm, header=None, index_col=0)

        for batch_idx, (X_batch, y_batch, image_filename) in enumerate(DataLoader(dataset, batch_size=1)):
            if isinstance(dpm, str):
                dpm_val = dpm_df.loc[image_filename].values[0]
            elif isinstance(dpm, Number) or dpm == False:
                dpm_val = dpm
            else:
                raise ValueError('dpm must be str, Number or False')

            # getting filter range
            if isinstance(filter, dict):
                filter_val = filter[image_filename[0]]
            else:
                filter_val = filter

            segmented_img = self.predict_single_large_image(X_batch, tile_res=tile_res)
            hypo_result_mask = segmented_img[:, :, 2]
            hypo_result, hypo_result_skeleton = get_hypo_rprops(hypo_result_mask, filter=filter_val,
                                                                return_skeleton=True, dpm=dpm_val)
            hypo_result.make_df().to_csv(os.path.join(export_path, image_filename[0][:-4] + '_result.csv'))

            if visualize_segmentation:
                io.imsave(os.path.join(export_path, image_filename[0][:-4] + '_segmentation_skeletons.png'),
                          img_as_uint(hypo_result_skeleton))
                io.imsave(os.path.join(export_path, image_filename[0][:-4] + '_segmentation_hypo.png'),
                          hypo_result_mask)
                io.imsave(os.path.join(export_path, image_filename[0][:-4] + '_segmentation_full.png'),
                          segmented_img)

            if not skeletonized_gt:
                hypo_gt_mask = y_batch[0].data.numpy() == 2
            else:
                hypo_gt_mask = y_batch[0].data.numpy() > 0

            hypo_result_gt = get_hypo_rprops(hypo_gt_mask, filter=[20/dpm_val, np.inf],
                                             already_skeletonized=skeletonized_gt, dpm=dpm_val)
            hypo_result_gt.make_df().to_csv(os.path.join(export_path, image_filename[0][:-4] + '_gt.csv'))

            scores[image_filename[0]], objectwise_df = hypo_result.score(hypo_result_gt,
                                                                         match_threshold=match_threshold)
            objectwise_df.to_csv(os.path.join(export_path, image_filename[0][:-4] + '_matched.csv'))

            # visualization
            # histograms
            if visualize_histograms:
                hypo_result.hist(hypo_result_gt,
                                 os.path.join(export_path, image_filename[0][:-4] + '_hist.png'))

            # bounding boxes
            if visualize_bboxes:
                visualize_regions(hypo_gt_mask, hypo_result_gt,
                                  export_path=os.path.join(export_path, image_filename[0][:-4] + '_gt.png'))
                visualize_regions(hypo_result_skeleton, hypo_result,
                                  export_path=os.path.join(export_path, image_filename[0][:-4] + '_result.png'))

        score_df = pd.DataFrame(scores).T
        score_df.to_csv(os.path.join(export_path, 'scores.csv'))

        return scores

    def visualize_workflow(self, dataset, export_path, filter=False):
        for image, mask, image_filename in DataLoader(dataset, batch_size=1):
            image_filename_root = image_filename[0].split('.')[0]
            hypo_full_result = self.predict_single_large_image(image, tile_res=(512, 512))
            hypo_result_mask = self.predict_single_large_image(image, channel=2, tile_res=(512, 512))
            # save multiclass mask
            io.imsave(os.path.join(export_path, image_filename_root + '_1.png'), hypo_full_result)
            hypo_result, hypo_result_skeleton = get_hypo_rprops(hypo_result_mask, return_skeleton=True, filter=filter)
            io.imsave(os.path.join(export_path, image_filename_root + '_2.png'), img_as_uint(hypo_result_skeleton))
            visualize_regions(image.data.cpu().numpy()[0].transpose((1, 2, 0)), hypo_result,
                              os.path.join(export_path, image_filename_root + '_3.png'))


if __name__ == '__main__':
    hypo_mask_img = io.imread('/home/namazu/Data/hypocotyl/measurement_test/masks/140925 8-4 050.png')
    hypo_result = get_hypo_rprops(hypo_mask_img, filter=False, already_skeletonized=True)
    hypo_result.score(hypo_result)
