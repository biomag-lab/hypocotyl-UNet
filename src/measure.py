import torch

from argparse import ArgumentParser

from unet.utils import *
from unet.unet import UNet

parser = ArgumentParser()
parser.add_argument("--images_path", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--result_folder", type=str, required=True)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--min_object_size", type=float, default=50)
parser.add_argument("--max_object_size", type=float, default=np.inf)

args = parser.parse_args()

predict_dataset = ReadTestDataset(args.images_path)
device = torch.device(args.device)

unet = UNet(3, 3)
unet.load_state_dict(torch.load(args.model, map_location=device))
model = ModelWrapper(unet, args.result_folder, cuda_device=device)
model.measure_large_images(predict_dataset, export_path=args.result_folder,
                           visualize_bboxes=False, filter=[args.min_object_size, args.max_object_size])
