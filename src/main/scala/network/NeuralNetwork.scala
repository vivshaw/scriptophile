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

	def feedForward (activation: DenseMatrix[Double]) : DenseMatrix[Double] = {
		var output = activation
		biases.zip(weights).foreach{ case (bias, weight) => 
			output = sigmoid((weight * output) + bias)
		}
		return output
	}

	def sgd (trainingData: List[Tuple2[DenseMatrix[Double], Int]], epochs: Int, miniBatchSize: Int, eta: Double) {
		val n = trainingData.length
		for(i <- 1 to epochs) {
			val data = shuffle(trainingData)
			val miniBatches = for (j <- 0 to n by miniBatchSize) yield trainingData.slice(j, j + miniBatchSize)
			miniBatches.foreach{ miniBatch => updateMiniBatch(miniBatch, eta) }
			println("Epoch " + i + " complete.")
		}
	}

	def updateMiniBatch (miniBatch: List[Tuple2[DenseMatrix[Double], Int]], eta: Double) {
		var nabla_bias = for (bias <- biases) yield DenseMatrix.zeros[Double](bias.rows, bias.cols)
		var nabla_weight = for (weight <- weights) yield DenseMatrix.zeros[Double](weight.rows, weight.cols)
		miniBatch.foreach{ case (features, result) =>
			val (delta_nabla_bias, delta_nabla_weight) = backprop(features, result)
			nabla_bias = for (t <- nabla_bias zip delta_nabla_bias) yield t._1 + t._2
			nabla_weight = for (t <- nabla_weight zip delta_nabla_weight) yield t._1 + t._2	
		}
		weights = for(t <- weights zip nabla_weight) yield t._1 - (t._2 * (eta / miniBatch.length))
		biases = for(t <- biases zip nabla_bias) yield t._1 - (t._2 * (eta / miniBatch.length))
	}

	def backprop (features: DenseMatrix[Double], result: Int) : (List[DenseMatrix[Double]], List[DenseMatrix[Double]]) = {
		var nabla_bias = for (bias <- biases) yield DenseMatrix.zeros[Double](bias.rows, bias.cols)
		var nabla_weight = for (weight <- weights) yield DenseMatrix.zeros[Double](weight.rows, weight.cols)
		return (nabla_bias, nabla_weight)
	}
}

object NeuralNetwork {
	def apply(x: List[Int]) = new NeuralNetwork(x)
}