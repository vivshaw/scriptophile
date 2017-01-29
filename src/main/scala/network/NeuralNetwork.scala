/**
  * Created by vivshaw on 1/28/2017.
  */

package network

import breeze.linalg._
import breeze.numerics._
import scala.util.Random.shuffle

class NeuralNetwork(sizes: List[Int]) {
  val layers = sizes.length
  val normal = breeze.stats.distributions.Gaussian(0, 1)
  var biases = for (y <- sizes.drop(1) ) yield DenseMatrix.rand(y, 1, normal)
  var weights = for (t <- sizes.dropRight(1) zip sizes.drop(1)) yield DenseMatrix.rand(t._2, t._1, normal)
  var test: Double = 0

  def feedForward (a: DenseMatrix[Double]) : DenseMatrix[Double] = {
  	var result = a
  	biases.zip(weights).foreach{ case (bias, weight) => 
  		result = sigmoid((weight * result) + bias)
  	}
  	return result
  }

  def sgd (trainingData: List[Tuple2[DenseMatrix[Double], Int]], epochs: Int, miniBatchSize: Int, eta: Double) {
  	val n = trainingData.length
  	for(i <- 1 to epochs) {
  		val data = shuffle(trainingData)
  		val miniBatches = for (k <- 0 to n by miniBatchSize) yield trainingData.slice(k, k + miniBatchSize)
  		miniBatches.foreach{ miniBatch => updateMiniBatch(miniBatch, eta) }
  	}
  }

  def updateMiniBatch (miniBatch: List[Tuple2[DenseMatrix[Double], Int]], eta: Double) {
  	test = eta
  	return eta
  }
}

object NeuralNetwork {
	def apply(x: List[Int]) = new NeuralNetwork(x)
}