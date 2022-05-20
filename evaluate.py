import time
import os

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from data import datasets
from model import loader
from metrics import AverageMeter, Result
from data import transforms

max_depths = {
    'kitti': 80.0,
    'nyu' : 10.0,
    'nyu_reduced' : 10.0,
}
nyu_res = {
    'full' : (480, 640),
    'half' : (240, 320),
    'mini' : (224, 224)}
kitti_res = {
    'full' : (384, 1280),
    'tu_small' : (128, 416),
    'tu_big' : (228, 912),
    'half' : (192, 640)}
resolutions = {
    'nyu' : nyu_res,
    'nyu_reduced' : nyu_res,
    'kitti' : kitti_res}
crops = {
    'kitti' : [128, 381, 45, 1196],
    'nyu' : [20, 460, 24, 616],
    'nyu_reduced' : [20, 460, 24, 616]}

class Evaluater():
    def __init__(self, args):
        self.debug = True
        self.dataset = args.dataset

        self.maxDepth = max_depths[args.dataset]
        self.res_dict = resolutions[args.dataset]
        self.resolution = self.res_dict[args.resolution]
        print('Resolution for Eval: {}'.format(self.resolution))
        self.resolution_keyword = args.resolution
        print('Maximum Depth of Dataset: {}'.format(self.maxDepth))
        self.crop = crops[args.dataset]
        self.eval_mode = args.eval_mode

        self.result_dir = args.save_results
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = loader.load_model(args.model, args.weights_path)
        self.model.to(self.device)

        self.test_loader = datasets.get_dataloader(args.dataset,
                                                 path=args.test_path,
                                                 split='test',
                                                 batch_size=1,
                                                 augmentation=args.eval_mode,
                                                 resolution=args.resolution,
                                                 workers=args.num_workers)

        self.downscale_image = torchvision.transforms.Resize(self.resolution) #To Model resolution

        self.to_tensor = transforms.ToTensor(test=True, maxDepth=self.maxDepth)


        self.visualize_images = [0, 1, 2, 3, 4, 5,
                                 100, 101, 102, 103, 104, 105,
                                 200, 201, 202, 203, 204, 205,
                                 300, 301, 302, 303, 304, 305,
                                 400, 401, 402, 403, 404, 405,
                                 500, 501, 502, 503, 504, 505,
                                 600, 601, 602, 603, 604, 605]

    def evaluate(self):
        self.model.eval()
        average_meter = AverageMeter()
        for i, data in enumerate(self.test_loader):
            t0 = time.time()
            image, gt = data
            packed_data = {'image': image[0], 'depth':gt[0]}
            data = self.to_tensor(packed_data)
            image, gt = self.unpack_and_move(data)
            image = image.unsqueeze(0)
            gt = gt.unsqueeze(0)

            image_flip = torch.flip(image, [3])
            gt_flip = torch.flip(gt, [3])
            if self.eval_mode == 'alhashim':
                # For model input
                image = self.downscale_image(image)
                image_flip = self.downscale_image(image_flip)

            data_time = time.time() - t0

            t0 = time.time()

            inv_prediction = self.model(image)
            prediction = self.inverse_depth_norm(inv_prediction)

            inv_prediction_flip = self.model(image_flip)
            prediction_flip = self.inverse_depth_norm(inv_prediction_flip)

            gpu_time = time.time() - t0

            if self.eval_mode == 'alhashim':
                upscale_depth = torchvision.transforms.Resize(gt.shape[-2:]) #To GT res

                prediction = upscale_depth(prediction)
                prediction_flip = upscale_depth(prediction_flip)

                if self.dataset == 'kitti':


                    gt_height, gt_width = gt.shape[-2:] 

                    self.crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,
                                          0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

                if i in self.visualize_images:
                    self.save_image_results(image, gt, prediction, i)

                gt = gt[:,:, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
                gt_flip = gt_flip[:,:, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
                prediction = prediction[:,:, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
                prediction_flip = prediction_flip[:,:, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]



            result = Result()
            result.evaluate(prediction.data, gt.data)
            average_meter.update(result, gpu_time, data_time, image.size(0))

            result_flip = Result()
            result_flip.evaluate(prediction_flip.data, gt_flip.data)
            average_meter.update(result_flip, gpu_time, data_time, image.size(0))

        #Report 
        avg = average_meter.average()
        current_time = time.strftime('%H:%M', time.localtime())
        self.save_results(avg)
        print('\n*\n'
              'RMSE={average.rmse:.3f}\n'
              'MAE={average.mae:.3f}\n'
              'Delta1={average.delta1:.3f}\n'
              'Delta2={average.delta2:.3f}\n'
              'Delta3={average.delta3:.3f}\n'
              'REL={average.absrel:.3f}\n'
              'Lg10={average.lg10:.3f}\n'
              't_GPU={time:.3f}\n'.format(
              average=avg, time=avg.gpu_time))

    def save_results(self, average):
        results_file = os.path.join(self.result_dir, 'results.txt')
        with open(results_file, 'w') as f:
            f.write('RMSE,MAE,REL, RMSE_log,Lg10,Delta1,Delta2,Delta3\n')
            f.write('{average.rmse:.3f}'
                    ',{average.mae:.3f}'
                    ',{average.absrel:.3f}'
                    ',{average.rmse_log:.3f}'
                    ',{average.lg10:.3f}'
                    ',{average.delta1:.3f}'
                    ',{average.delta2:.3f}'
                    ',{average.delta3:.3f}'.format(
                        average=average))


    def inverse_depth_norm(self, depth):
        depth = self.maxDepth / depth
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        return depth


    def depth_norm(self, depth):
        depth = torch.clamp(depth, self.maxDepth / 100, self.maxDepth)
        depth = self.maxDepth / depth
        return depth


    def unpack_and_move(self, data):
        if isinstance(data, (tuple, list)):
            image = data[0].to(self.device, non_blocking=True)
            gt = data[1].to(self.device, non_blocking=True)
            return image, gt
        if isinstance(data, dict):
            keys = data.keys()
            image = data['image'].to(self.device, non_blocking=True)
            gt = data['depth'].to(self.device, non_blocking=True)
            return image, gt
        print('Type not supported')

    def save_image_results(self, image, gt, prediction, image_id):
        img = image[0].permute(1, 2, 0).cpu()
        gt = gt[0,0].permute(0, 1).cpu()
        prediction = prediction[0,0].permute(0, 1).detach().cpu()
        error_map = gt - prediction
        vmax_error = self.maxDepth / 10.0
        vmin_error = 0.0
        cmap = 'viridis'

        vmax = torch.max(gt[gt != 0.0])
        vmin = torch.min(gt[gt != 0.0])

        save_to_dir = os.path.join(self.result_dir, 'image_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)
        fig.savefig(save_to_dir)
        plt.clf()

        save_to_dir = os.path.join(self.result_dir, 'errors_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        errors = ax.imshow(error_map, vmin=vmin_error, vmax=vmax_error, cmap='Reds')
        fig.colorbar(errors, ax=ax, shrink=0.8)
        fig.savefig(save_to_dir)
        plt.clf()

        save_to_dir = os.path.join(self.result_dir, 'gt_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(gt, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.savefig(save_to_dir)
        plt.clf()

        save_to_dir = os.path.join(self.result_dir, 'depth_{}.png'.format(image_id))
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(prediction, vmin=vmin, vmax=vmax, cmap=cmap)
        fig.savefig(save_to_dir)
        plt.clf()
