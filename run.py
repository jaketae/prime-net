import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models import PriSTM


def main(args):
    model = PriSTM(
        args.embed_dim, args.hidden_size, args.num_layers, args.bidirectional
    )
    trainer = Trainer(gpus=args.gpus, callbacks=[EarlyStopping(monitor="f1")])
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--gpus", type=int, default=None)
    args = parser.parse_args()
    main(args)
