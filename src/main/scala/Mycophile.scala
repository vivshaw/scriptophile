/**
  * Created by vivshaw on 1/28/2017.
  */
package mycophile

import breeze.linalg._
import breeze.numerics._

import network.NeuralNetwork

object Mycophile extends App {
	println("Hello, Scala!")
	var net = NeuralNetwork(List(20, 8, 1))
	val test_matrix = DenseMatrix.zeros[Double](20,1)
	println(net.feedforward(test_matrix))
}