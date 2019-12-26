import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim

from unet.unet import UNet
from unet.utils import *


parser = ArgumentParser()
parser.add_argument("--train_dataset", type=str, required=True)
parser.add_argument("--val_dataset", type=str, default=None)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--model_name", type=str, default="UNet-hypocotyl")
parser.add_argument("--trained_model_path", type=str, default=None)
parser.add_argument("--initial_lr", type=float, default=1e-3)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--model_save_freq", type=int, default=200)

args = parser.parse_args()

tf_train = make_transform(crop=(512, 512), long_mask=True)
tf_validate = make_transform(crop=(512, 512), long_mask=True, rotate_range=False,
                             p_flip=0.0, normalize=False, color_jitter_params=None)

# load dataset
train_dataset_path = args.train_dataset
train_dataset = ReadTrainDataset(train_dataset_path, transform=tf_train)

if args.val_dataset is not None:
    validate_dataset_path = args.val_dataset
    validate_dataset = ReadTrainDataset(validate_dataset_path, transform=tf_validate)
else:
    validate_dataset = None

tf_train = make_transform(crop=(512, 512), long_mask=True, p_random_affine=0.0)
tf_validate = make_transform(crop=(512, 512), long_mask=True, rotate_range=False,
                             p_flip=0.0, normalize=False, color_jitter_params=None)

# creating checkpoint folder
model_name = args.model_name
file_dir = os.path.split(os.path.realpath(__file__))[0]
results_folder = os.path.join(file_dir, '..', 'checkpoints', model_name)
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# load model
unet = UNet(3, 3)
if args.trained_model_path is not None:
    unet.load_state_dict(torch.load(args.trained_model_path))

loss = SoftDiceLoss(weight=torch.Tensor([1, 5, 5]))
optimizer = optim.Adam(unet.parameters(), lr=args.initial_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, verbose=True)

cuda_device = torch.device(args.device)
model = ModelWrapper(unet, loss=loss, optimizer=optimizer, scheduler=scheduler,
                     results_folder=results_folder, cuda_device=cuda_device)

model.train_model(train_dataset, validation_dataset=validate_dataset,
                  n_batch=args.batch_size, n_epochs=args.epochs,
                  verbose=False, save_freq=args.model_save_freq)
