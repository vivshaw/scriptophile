# scriptophile
a neural network trained to read handwritten digits, written from scratch in Scala

## What It Is
This project implements a simple neural network, then trains it using mini-batch stochastic gradient descent on the [MNIST handwritten digit dataset](http://yann.lecun.com/exdb/mnist/). No fancy machine learning libraries here, this is written from scratch in plain Scala. I'm using [Breeze](https://github.com/scalanlp/breeze) for matrix algebra, and that's about it.

The project was inspired by [Michael Nielsen](http://michaelnielsen.org/)'s excellent textbook [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/), and my code is effectively a Scala port of [Nielsen's algorithms](https://github.com/mnielsen/neural-networks-and-deep-learning).

## How To Use It
```
$git clone https://github.com/vivshaw/scriptophile.git
$cd scriptophile
$sbt run
```

## Typical Results

I typically get recognition accuracy in the 94%-95% range. A typical run looks like this:

```
$sbt run
now training
Epoch 1 complete, with 7969 / 10000 correct
Epoch 2 complete, with 8162 / 10000 correct
Epoch 3 complete, with 8246 / 10000 correct
Epoch 4 complete, with 9232 / 10000 correct
...
Epoch 29 complete, with 9388 / 10000 correct
Epoch 30 complete, with 9402 / 10000 correct
final accuracy: 0.9402%
```

## To-dos

* Rewrite in more idiomatic Scala
* Hunt for suspected implementation error causing poor recognition accuracy with large hidden layers
