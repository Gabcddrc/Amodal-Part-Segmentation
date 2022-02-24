import comet_ml 
import torch
from model import ModelWrapper
from tqdm import tqdm
import utils
import vis
from utils import push_images, log_dict, Experiment
import time
import numpy as np
import os.path as op


"""
The Trainer class abstracts aways lots of boilerplate code such as saving weights to disks, training epochs, evaluation epochs, visualization etc.
Here is the pseudo-code of how each method interacts

def train:
    for each epoch:
        train_epoch() # train one epoch
        if every k epochs:
            eval_epoch() # eval on validation set
            log_dict() # logging performance
        save_model()

See self.train() and its subroutines for details
"""

class Trainer():
    def __init__(self, train_loader, val_loaders, args):
        super().__init__()
        self.args = args
        self.train_loader = train_loader
        self.val_loaders = val_loaders

        self.model = ModelWrapper()
        if self.args.load_ckpt != '':
            self.load_ckpt(self.args.load_ckpt)

        self.model.cuda()
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=args.lr) 

        self.current_epoch = 0
        self.global_step = 0

        # experiment key
        if self.args.load_ckpt == '':
            self.experiment = Experiment(args)
            self.args.exp_key = self.experiment.get_key()[:9]
        else:
            self.experiment =  Experiment(args)
            self.args.exp_key = self.args.load_ckpt.split('/')[1]
        print('Experiment Key: %s' % (self.args.exp_key))

        # folder containing info of this experiment
        self.exp_path = op.join('logs', self.args.exp_key)
        self.save_path = op.join(self.exp_path, 'latest.pt')  # model dumping
        utils.mkdir_p(self.exp_path)

    def train_epoch(self):
        assert self.train_loader is not None
        model = self.model
        train_loader = self.train_loader
        optimizer = self.optimizer
        epoch = self.current_epoch

        model.train()
        print_every = 20
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader))

        # training loop for epoch
        for i, batch in pbar:
            # push things to CUDA
            inputs, targets, meta_info = utils.things2dev(batch, 'cuda')

            # standard pytorch boilerplate stuff
            optimizer.zero_grad()
            loss = model(inputs, targets, meta_info, 'train')
            total_loss = loss['loss_segm']
            total_loss.backward()
            optimizer.step()

            # logging
            running_loss += total_loss.item()
            if i % print_every == print_every - 1:
                avg_loss = running_loss / print_every
                pbar.set_description(
                    'Epoch=%d loss=%.5f' %
                    (epoch + 1, avg_loss))
                if self.experiment is not None:
                    self.experiment.log_metric(
                            'total_loss', avg_loss, self.global_step)
                running_loss = 0.0
            self.global_step += 1

        self.current_epoch += 1

    def eval_epoch(self, val_loader_dict):
        # evaluate on a data loader
        assert isinstance(val_loader_dict, dict)
        model = self.model
        val_loader = val_loader_dict['loader']
        postfix = val_loader_dict['postfix']

        pbar = tqdm(enumerate(val_loader, 0), total=len(val_loader))
        out_list = []
        model.eval()
        with torch.no_grad():
            for i, batch in pbar:
                inputs, targets, meta_info = utils.things2dev(
                    batch, 'cuda')
                out = model(inputs, targets, meta_info, 'val')
                out_list.append(out)

        # aggregate outputs from each batch
        out_dict = utils.ld2dl(utils.things2dev(out_list, 'cpu'))

        miou = np.nanmean(np.concatenate(out_dict['ious'], axis=0))
        metric_dict = {'miou_' + postfix: miou}
        return metric_dict

    def load_ckpt(self, ckpt_path):
        sd = torch.load(ckpt_path)
        print(self.model.load_state_dict(sd))

    def test_epoch(self, val_loader):
        assert val_loader is not None
        model = self.model

        pbar = tqdm(enumerate(val_loader, 0), total=len(val_loader))
        out_list = []
        model.eval()
        with torch.no_grad():
            for i, batch in pbar:
                inputs, meta_info = utils.things2dev(
                    batch, 'cuda')
                # forward without providing segmentation
                out = model.forward_test(inputs, meta_info)
                out_list.append(out)

        # aggregate segmentation predictions
        out_dict = utils.ld2dl(utils.things2dev(out_list, 'cpu'))
        segm_l = torch.cat(out_dict['segm_l'], dim=0)
        segm_r = torch.cat(out_dict['segm_r'], dim=0)
        im_path = sum(out_dict['im_path'], [])
        return (segm_l, segm_r, im_path)

    def test(self, val_loader):
        # evaluate a model in the test set. See test.py
        # package the prediction into test.tar.gz, which is used for submission
        segm_l, segm_r, im_path = self.test_epoch(val_loader)
        out_test_path = op.join(self.exp_path, 'test.lmdb')
        tar_path = out_test_path + '.tar.gz'

        utils.package_lmdb(out_test_path, segm_l, segm_r, im_path)
        im_path = [imp + '.jpg' for imp in im_path]
        torch.save(im_path, out_test_path + '/im_path.pt')
        print('Done writing test to: %s' % (out_test_path))

        utils.make_tarfile(tar_path, out_test_path)
        print('Done zipping test to: %s' % (tar_path))

    def visualize_batches(self, batch, postfix, num_examples, no_tqdm=True):
        # visualize a given batch

        model = self.model
        im_list = []
        model.eval()

        tic = time.time()
        with torch.no_grad():
            inputs, targets, meta_info = utils.things2dev(
                    batch, 'cuda')

            # get objects for visualizaton
            vis_dict = model(inputs, targets, meta_info, 'vis')

            # visualization plots
            curr_im_list = vis.visualize_all(
                    vis_dict, num_examples,
                    postfix=postfix, no_tqdm=no_tqdm)

            # push images to logger
            push_images(
                    self.experiment, curr_im_list, self.global_step)
            im_list += curr_im_list

        print('Done rendering (%.1fs)' % (time.time() - tic))
        return im_list

    def save_model(self):
        sd = self.model.state_dict()
        sd = utils.things2dev(sd, 'cpu')
        torch.save(sd, self.save_path)
        print('Saved model to: %s' % (self.save_path))

    def train(self):
        for epoch in range(self.args.num_epoch):
            # train one epoch
            self.train_epoch()
            if self.current_epoch % self.args.eval_every_epoch == 0:
                # evaluate on a list of loaders
                for loader in self.val_loaders:
                    # metric performance on each loader
                    metric_dict = self.eval_epoch(loader)

                    # push metrics to logger
                    log_dict(self.experiment, metric_dict, self.global_step)

                    # visualize loader batch
                    self.visualize_batches(
                            loader['vis_batch'], postfix=loader['postfix'], num_examples=20)
                    print(metric_dict)
                self.save_model()
        print('Finished Training')
