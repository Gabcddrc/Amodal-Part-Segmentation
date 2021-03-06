import trainer as tr
import datasets
from pprint import pprint
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from datasets import ImageDataset


def run_exp(args):
    # training loader
    # size of training set depends on --trainsplit
    train_loader = datasets.fetch_dataloader(
            'train', args.trainsplit, args)

    # evaluation loader to see training set performance
    # Use minitrain (1000 samples)
    train_val_loader = {
            'loader': datasets.fetch_dataloader('train', 'minitrain', args),
            'postfix': "__train"}
    train_val_loader['vis_batch'] = next(iter(train_val_loader['loader'])) # samples to visualize

    # evaluation loader to see validation set performance
    # size of validataion set depends on --valsplit
    val_val_loader = {
            'loader': datasets.fetch_dataloader('val', args.valsplit, args),
            'postfix': "__val"}
    val_val_loader['vis_batch'] = next(iter(val_val_loader['loader']))

    # unpaired images for training
    #unpaired_images_loader = datasets.fetch_dataloader('images', None, args)

    # unpaired segmentation for training
    #unpaired_segm_loader = datasets.fetch_dataloader('segm', None, args)

    trainer = tr.Trainer(train_loader, [train_val_loader, val_val_loader], args)
    trainer.train()


def run_full_exp(args):
    """
    Training with both training set and validation sets for final submission.
    """

    trainset = ImageDataset(transforms.ToTensor(), 'train', 'train', args)
    valset = ImageDataset(transforms.ToTensor(), 'val', 'val', args)
    dataset = ConcatDataset([trainset, valset])
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True,
        drop_last=True)

    trainer = tr.Trainer(loader, [], args)
    trainer.train()


if __name__ == "__main__":
    from config import args
    pprint(vars(args))
    if args.trainsplit == 'trainval':
        run_full_exp(args)
    else:
        run_exp(args)
