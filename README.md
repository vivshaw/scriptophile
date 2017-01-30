# scriptophile
a neural network trained to read handwritten digits, written from scratch in Scala

## What It Is
This project implements a simple neural network, then trains it using mini-batch stochastic gradient descent on the [MNIST handwritten digit dataset](http://yann.lecun.com/exdb/mnist/). No fancy machine learning libraries here, this is written from scratch in plain Scala! I'm using [Breeze](https://github.com/scalanlp/breeze) for matrix algebra, and that's about it.

The project was inspired by [Michael Nielsen](http://michaelnielsen.org/)'s excellent textbook [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/), and my code is effectively a Scala port of [Nielsen's algorithms](https://github.com/mnielsen/neural-networks-and-deep-learning).

## How To Use It
```
git clone https://github.com/vivshaw/scriptophile.git
cd scriptophile
sbt run
```
