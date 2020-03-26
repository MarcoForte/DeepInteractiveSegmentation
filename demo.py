import argparse
from networks.transforms import trimap_transform, groupnorm_normalise_image
from networks.models import build_model

import numpy as np
import cv2
import torch
from dataloader import AlphaTestDataset
from interaction import robot_click, jaccard, remove_non_fg_connected


def NOCS(ious, thresh):
    nocs = []
    for i in range(ious.shape[0]):
        for j in range(20):
            if(ious[i, j] >= thresh):
                nocs.append(j + 1)
                break
        if(len(nocs) == i):
            nocs.append(20)
    return nocs


def np_to_torch(x):
    return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()


def scale_input(x: np.ndarray, scale_type) -> np.ndarray:
    ''' Scales so that min side length is 352 and sides are divisible by 8'''
    h, w = x.shape[:2]
    h1 = int(np.ceil(h / 32) * 32)
    w1 = int(np.ceil(w / 32) * 32)
    x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
    return x_scale


def pred(image_np: np.ndarray, trimap_np: np.ndarray, alpha_old_np: np.ndarray, model) -> np.ndarray:
    ''' Predict segmentation
        Parameters:
        image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        trimap_np -- two channel trimap/Click map, first background then foreground. Dimensions: (h, w, 2)
        Returns:
        alpha: alpha matte/non-binary segmentation image between 0 and 1. Dimensions: (h, w)
    '''
    # return trimap_np[:,:,1] + (1-np.sum(trimap_np,-1))/2
    alpha_old_np = remove_non_fg_connected(alpha_old_np, trimap_np[:, :, 1])

    h, w = trimap_np.shape[:2]
    image_scale_np = scale_input(image_np, cv2.INTER_LANCZOS4)
    trimap_scale_np = scale_input(trimap_np, cv2.INTER_NEAREST)
    alpha_old_scale_np = scale_input(alpha_old_np, cv2.INTER_LANCZOS4)

    with torch.no_grad():

        image_torch = np_to_torch(image_scale_np)
        trimap_torch = np_to_torch(trimap_scale_np)
        alpha_old_torch = np_to_torch(alpha_old_scale_np[:, :, None])

        trimap_transformed_torch = np_to_torch(trimap_transform(trimap_scale_np))
        image_transformed_torch = groupnorm_normalise_image(image_torch.clone(), format='nchw')

        alpha = model(image_transformed_torch, trimap_transformed_torch, alpha_old_torch, trimap_torch)
        alpha = cv2.resize(alpha[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)
    alpha[trimap_np[:, :, 0] == 1] = 0
    alpha[trimap_np[:, :, 1] == 1] = 1

    alpha = remove_non_fg_connected(alpha, trimap_np[:, :, 1])
    return alpha


def test(model, args):
    test_dset = AlphaTestDataset(args.dataset_dir)
    ious = np.zeros((test_dset.__len__(), args.num_clicks))

    for i in range(ious.shape[0]):
        item_dict = test_dset.__getitem__(i)
        image = item_dict['image']
        gt = item_dict['alpha']
        name = item_dict['name']
        h, w = gt.shape
        trimap = np.zeros((h, w, 2))
        alpha = np.zeros((h, w))
        for j in range(ious.shape[1]):
            trimap, click_region, [y, x], click_cat = robot_click(alpha >= 0.5, gt, trimap)
            alpha = pred(image, trimap, alpha, model)
            ious[i, j] = jaccard(gt == 1, alpha >= 0.5, np.abs(gt - 0.5) < 0.25)

            if(args.predictions_dir != ''):
                cv2.imwrite(f'{args.predictions_dir}/{name}_{i}_{j+1}.png', alpha * 255)

    nocs_90 = NOCS(ious, 0.9)
    mIoU = np.mean(ious)
    print(f'Average number of clicks to reach 90% {np.mean(nocs_90)} {nocs_90}')
    print(f'Mean IoU {mIoU}')


if __name__ == '__main__':

    def str2bool(v):
        # https://stackoverflow.com/a/43357954
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--use_mask_input', type=str2bool, nargs='?', const=True, default=True, help='')
    parser.add_argument('--use_usr_encoder', type=str2bool, nargs='?', const=True, default=True, help='')
    parser.add_argument('--weights', default='InterSegSynthFT.pth', help="pytorch state dict")

    # Evaluation related arguments
    parser.add_argument('--iou_lim', default=None, type=float, help='iou lim')
    parser.add_argument('--dataset_dir', default='./GrabCut/', help='dataset to test on')

    parser.add_argument('--predictions_dir', default='', help='Where to store predictions, if blank '' dont save ')
    parser.add_argument('--num_clicks', default=20, type=int, help='Number of clicks per image')
    args = parser.parse_args()
    model = build_model(args)
    model.eval()

    test(model, args)
