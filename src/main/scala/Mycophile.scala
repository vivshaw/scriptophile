/**
  * Created by vivshaw on 1/28/2017.
  */
package mycophile

import breeze.linalg._
import breeze.numerics._

import network.NeuralNetwork

object Mycophile extends App {
	var net = NeuralNetwork(List(2, 3, 2))
	val false_vector = new DenseMatrix(2,1,DenseVector(0.0, 1.0).toArray)
	val true_vector = new DenseMatrix(2,1,DenseVector(1.0, 0.0).toArray)
	val both = new DenseMatrix(2,1,DenseVector(1.0, 1.0).toArray)
	val neither = new DenseMatrix(2,1,DenseVector(0.0, 0.0).toArray)
	val train_data = List((both, true_vector), (neither, false_vector))

	println("pre-train evaluation: ")
	println(net.evaluate(train_data) + "/" + train_data.length + " correct")

	println("now training")
	net.sgd(train_data, 10000, 2, 0.5)

	println("post-train evaluation: ")
	println(net.evaluate(train_data) + "/" + train_data.length + " correct")
}