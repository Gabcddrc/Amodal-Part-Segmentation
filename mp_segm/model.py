import torch
import torch.nn as nn
import loss
import torch.nn.functional as F
import numpy as np
import utils
import unet_parts as parts
from unet_ae import UnetResnetAE

class UNet(nn.Module):
    # Source: https://github.com/milesial/Pytorch-UNet
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = parts.DoubleConv(n_channels, 64)
        self.down1 = parts.Down(64, 128)
        self.down2 = parts.Down(128, 256)
        self.down3 = parts.Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = parts.Down(512, 1024 // factor)
        self.up1 = parts.Up(1024, 512 // factor, bilinear)
        self.up2 = parts.Up(512, 256 // factor, bilinear)
        self.up3 = parts.Up(256, 128 // factor, bilinear)
        self.up4 = parts.Up(128, 64, bilinear)
        self.outc = parts.OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits



class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.segm_model = UnetResnetAE(in_channels = 3, num_classes =17*2, backend='resnet50')

      

    def forward(self, img):
        batch_size, _, im_dim, _ = img.shape

        # segmentation logits for prediction
        # amodal_logits = self.segm_model(img)['out']
        amodal_logits = self.segm_model(img)
        # split segmentation logits for left and right hands
        amodal_logits_l, amodal_logits_r = torch.split(amodal_logits, 17, 1)

        # convert logits to classes
        segm_mask_l = self.map2labels(amodal_logits_l)
        segm_mask_r = self.map2labels(amodal_logits_r)

        segm_dict = {}
        segm_dict['segm_mask_l'] = segm_mask_l  # segmentation classes
        segm_dict['segm_mask_r'] = segm_mask_r
        segm_dict['segm_logits'] = amodal_logits  # logits for the segmentaion

        out_dict = {}
        out_dict['segm_dict'] = segm_dict
        return out_dict

    def map2labels(self, segm_hand):
        # convert segmentation logits to labels
        with torch.no_grad():
            segm_hand = segm_hand.permute(0, 2, 3, 1)
            # class with max response
            _, pred_segm_hand = segm_hand.max(dim=3)
            return pred_segm_hand


class ModelWrapper(nn.Module):
    def __init__(self):
        super(ModelWrapper, self).__init__()

        # modules
        self.model = Model()

        # loss functions
        self.segm_loss = loss.SegmLoss()

    def forward_test(self, inputs, meta_info):
        # this forward function is used for test set
        # as the test set does not contain segmentation annotation
        # please be careful when changing this function in case it breaks packaging code for submission
        input_img = inputs['img']
        model_dict = self.model(input_img)
        segm_dict = model_dict['segm_dict']

        # images involved in this test set
        im_path = meta_info['im_path']
        im_path = [p.replace('./data/InterHand2.6M/images/', '').replace('.jpg', '') for p in im_path]

        # predictions
        return {'segm_l': segm_dict['segm_mask_l'],
                'segm_r': segm_dict['segm_mask_r'],
                'im_path': im_path}

    def forward(self, inputs, targets, meta_info, mode):
        loss_dict = {}

        # unpacking
        input_img = inputs['img']
        segm_target_128 = targets['segm_128']

        # prediction
        model_dict = self.model(input_img)
        segm_dict = model_dict['segm_dict']
        segm_dict['segm_128'] = segm_target_128

        # FS loss
        loss_dict['loss_segm'] = self.segm_loss(segm_dict)

        if mode == 'train':
            # if training, return loss
            return loss_dict

        if mode == 'vis':
            # if visualization, return vis_dict containing objects for visualization
            vis_dict = {}

            segm_l_mask_128 = F.interpolate(
                    segm_dict['segm_mask_l'].float()[:, None, :, :], 128, mode='nearest').long().squeeze()
            segm_r_mask_128 = F.interpolate(
                    segm_dict['segm_mask_r'].float()[:, None, :, :], 128, mode='nearest').long().squeeze()

            # packaging for visualization
            vis_dict['segm_l_mask'] = utils.tensor2np(segm_l_mask_128)
            vis_dict['segm_r_mask'] = utils.tensor2np(segm_r_mask_128)
            vis_dict['input_img'] = utils.tensor2np(input_img)

            # segmentation groundtruth
            vis_dict['segm_target_128'] = utils.tensor2np(
                    F.interpolate(segm_target_128.float(), 128, mode='nearest').long())
            vis_dict['im_path'] = meta_info['im_path']
            return vis_dict

        # for evaluation
        loss_dict = {k: loss_dict[k].mean() for k in loss_dict}
        loss_dict['total_loss'] = sum(loss_dict[k] for k in loss_dict)

        # predictions
        segm_l_mask_128 = segm_dict['segm_mask_l']
        segm_r_mask_128 = segm_dict['segm_mask_r']

        # GT
        segm_target = segm_target_128
        segm_target_l = segm_target[:, 0]
        segm_target_r = segm_target[:, 1]

        # evaluation loop
        ious_l = []
        ious_r = []
        for idx in range(segm_target.shape[0]):
            # Warning: do not modify these two lines
            iou_l = utils.segm_iou(segm_l_mask_128[idx], segm_target_l[idx], 17, 20)
            iou_r = utils.segm_iou(segm_r_mask_128[idx], segm_target_r[idx], 17, 20)
            ious_l.append(iou_l)
            ious_r.append(iou_r)

        out = {}
        out['loss'] = loss_dict
        out['ious'] = np.array(ious_l + ious_r)
        return out
