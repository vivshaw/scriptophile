/**
  * Created by vivshaw on 1/28/2017.
  */

package network

import breeze.linalg._
import breeze.numerics._

class NeuralNetwork(sizes: List[Int]) {
  val layers = sizes.length
  val normal = breeze.stats.distributions.Gaussian(0, 1)
  var biases = for (y <- sizes.drop(1) ) yield DenseMatrix.rand(y, 1, normal)
  var weights = for (t <- sizes.dropRight(1) zip sizes.drop(1)) yield DenseMatrix.rand(t._2, t._1, normal)
}

object NeuralNetwork {
	def apply(x: List[Int]) = new NeuralNetwork(x)
}