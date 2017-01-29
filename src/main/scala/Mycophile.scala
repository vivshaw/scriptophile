/**
  * Created by vivshaw on 1/28/2017.
  */
package mycophile

import breeze.linalg._
import breeze.numerics._

import network.NeuralNetwork

object Mycophile extends App {
	var net = NeuralNetwork(List(2, 2, 2))
	val zero = new DenseMatrix(2,1,DenseVector(0.0, 1.0).toArray)
	val one = new DenseMatrix(2,1,DenseVector(1.0, 0.0).toArray)
	val train_data = List((DenseMatrix.rand(2, 1, breeze.stats.distributions.Gaussian(0, 1)), zero), (DenseMatrix.rand(2, 1, breeze.stats.distributions.Gaussian(0, 1)), one))

	println("pre-train evaluation: ")
	println(net.evaluate(train_data) + "/" + train_data.length + " correct")

	println("now training")
	net.sgd(train_data, 50000, 1, 0.1)

	println("evaluation after training")
	println(net.evaluate(train_data) + "/" + train_data.length + " correct")

	println("datum 1")
	println(net.feedForward(train_data(0)._1))
	println(argmax(net.feedForward(train_data(0)._1)))

	println("datum 2")
	println(net.feedForward(train_data(1)._1))
	println(argmax(net.feedForward(train_data(1)._1)))

	println("post-train evaluation: ")
	println(net.evaluate(train_data) + "/" + train_data.length + " correct")
}