import trainer as tr
import datasets


def run_exp(args):
    train_loader = None
    # load test set
    loader = datasets.fetch_dataloader('test', 'test', args)

    # initialize network with boilerplate objects
    trainer = tr.Trainer(train_loader, [], args)

    # evaluate on the test set
    trainer.test(loader)


if __name__ == "__main__":
    from config import args
    run_exp(args)
