# Prime Net

This is a mini project in which I explore the feasability of detecting prime numbes through artifical neural networks, specifically RNNS.

## Motivation

Prime number detection and prediction are on-going endeavors in the mathematical circle. Communal projects such as the [Great Internet Mersenne Prime Search](https://www.mersenne.org) are manifestations of such efforts. Aside from the inherent intellectual value of discovering large primes, prime number detection has wide-ranging implications in various fields, such as crytograpy.

Artifical neural networks are known to be [universal function approximators](https://en.wikipedia.org/wiki/Universal_approximation_theorem). If we can consider the distribution of prime numbers as a step-wise or a [Dirac delta](https://en.wikipedia.org/wiki/Dirac_delta_function)-esque spike function, in which `f(x)` is 1 if `x` is prime and 0 otherwise, perhaps a neural network could be used to approximate it.

## Model

We implement a very simple Bidirectional LSTM network. To feed numbers into the model, we use binary encoding with padding. For instance,

```python
binary_encoding(10, width=8) = "10102222"
```

The first four digits are simply the number expressed in binary form, and the rest of the `2`s are padding tokens used to batch input numbers.

## Execution

The project uses [PyTorch Lightning](https://pytorchlightning.ai) to minimize boilerplate code. PyTorch Lightning's Trainer API is used to train, validate, and test the model, all in one go.

To execute the pipeline, type

```
python run.py
```

Below is the full list of additional arguments.

```
usage: run.py [-h] [--embed_dim EMBED_DIM] [--hidden_size HIDDEN_SIZE]
              [--num_layers NUM_LAYERS] [--bidirectional BIDIRECTIONAL]
              [--batch_size BATCH_SIZE] [--gpus GPUS]
```

## Result

Perhaps in large part due to the simplicity of the network, or more likely due to the absence of learnable features in the binary padded encoding, the LSTM network was not able to learn any significant patterns. Specifically, the model failed to accurately determine whether a given number was prime or not if the provided number was beyond the range of its training data.

## Future Directions

The following are some interesting questions worth further exploration.

-   Could using transformers improve recall?
-   Should numbers be padded with a dedicated padding token or simply 0?
-   What are the implications of setting `zero_division` to 0 or 1 in scikit-learn's metric calculation?
-   The current method of binary padding encoding can only express up to `2 ** width` numbers. What could be a length-invariant way of representing natural numbers?
