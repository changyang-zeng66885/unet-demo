import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='D:\\Pytorch-UNet-master\\Pytorch-UNet-master\\checkpoints', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT',nargs='+', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    # args = get_args()
    # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    #in_files = args.input
    
    in_files = ["test_data/imgs_png/frame_0.png","test_data/imgs_png/frame_1.png"]
    #out_files = get_output_filenames(args)
    # out_files = "test_data/predict_masks"
    out_files = ["test_data/masks_png/frame_0.png","test_data/masks_png/frame_1.png"]

    #net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    net = UNet(n_channels=1, n_classes=1, bilinear=False)

    model = "checkpoints/checkpoint_0719.pth"
    # model = "checkpoints/unet_carvana_scale1.0_epoch2.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logging.info(f'Loading model {args.model}')
    # logging.info(f'Using device {device}')
    print(f'Loading model {model}')
    print(f'Using device {device}')

    net.to(device=device)
    #state_dict = torch.load(args.model, map_location=device)
    state_dict = torch.load(model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    # logging.info('Model loaded!')
    print('Model loaded!')

    
    print("in_files:",in_files)
    for i, filename in enumerate(in_files):
        # logging.info(f'Predicting image {filename} ...')
        print(f'Predicting image {filename} ...')
        img = Image.open(filename)

        if len(img.getbands()) == 4 or len(img.getbands()) == 3:
            # 如果图像有 4 个通道或 3 个通道,则转换为单通道图像
            img = Image.fromarray(np.array(img)[:, :, 0]).convert('L')

        # mask = predict_img(net=net,
        #                    full_img=img,
        #                    scale_factor=args.scale,
        #                    out_threshold=args.mask_threshold,
        #                    device=device)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=1,
                           out_threshold=0.5,
                           device=device)

        # if not args.no_save:
        ## 保存结果
        out_filename = out_files[i]
        result = mask_to_image(mask, mask_values)
        result.save(out_filename)
        logging.info(f'Mask saved to {out_filename}')

        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, mask)
