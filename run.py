import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dataset import make_loader
from models import PriSTM


def main(args):
    model = PriSTM(
        args.embed_dim, args.hidden_size, args.num_layers, args.bidirectional
    )
    trainer = Trainer(
        gpus=args.gpus,
        progress_bar_refresh_rate=20,
        callbacks=[EarlyStopping(monitor="recall")],
    )
    batch_size = args.batch_size
    train_loader = make_loader("train", batch_size)
    val_loader = make_loader("val", batch_size)
    test_loader = make_loader("test", batch_size)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=None)
    args = parser.parse_args()
    main(args)
