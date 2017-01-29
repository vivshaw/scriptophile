/**
  * Created by vivshaw on 1/28/2017.
  */
package mycophile

import breeze.linalg._
import breeze.numerics._

import network.NeuralNetwork

object Mycophile extends App {
	var net = NeuralNetwork(List(2, 3, 2))

	val zero = new DenseMatrix(2,1,DenseVector(0.0, 1.0).toArray)
	val one = new DenseMatrix(2,1,DenseVector(1.0, 0.0).toArray)
	val both = new DenseMatrix(2,1,DenseVector(1.0, 1.0).toArray)
	val neither = new DenseMatrix(2,1,DenseVector(0.0, 0.0).toArray)

	val excl_or = List((zero, one), (one, one), (both, zero), (neither, zero))
	val incl_or = List((zero, one), (one, one), (both, one), (neither, zero))
	val train_data = excl_or

	println("now training")
	net.sgd(train_data, 10000, 4, 0.5, train_data)
}