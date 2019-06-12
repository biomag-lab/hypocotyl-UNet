import torch

from argparse import ArgumentParser

from unet.utils import *
from unet.unet import UNet

parser = ArgumentParser()
parser.add_argument("--images_path", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--result_folder", type=str, required=True)
parser.add_argument("--device", type=str, default='cpu')
parser.add_argument("--min_object_size", type=float, default=0)
parser.add_argument("--max_object_size", type=float, default=np.inf)
parser.add_argument("--dpi", type=float, default=False)
parser.add_argument("--dpm", type=float, default=False)
args = parser.parse_args()

# determining dpm
dpm = args.dpm if args.dpi else dpi_to_dpm(args.dpi)

print("Loading dataset...")
predict_dataset = ReadTestDataset(args.images_path)
device = torch.device(args.device)
print("Dataset loaded")

print("Loading model...")
unet = UNet(3, 3)
unet.load_state_dict(torch.load(args.model, map_location=device))
model = ModelWrapper(unet, args.result_folder, cuda_device=device)
print("Model loaded")
print("Measuring images...")
model.measure_large_images(predict_dataset, export_path=args.result_folder,
                           visualize_bboxes=False, filter=[args.min_object_size, args.max_object_size],
                           dpm=args.dpm, verbose=True)
