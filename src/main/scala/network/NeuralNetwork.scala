package network

import breeze.linalg._
import breeze.numerics._
import scala.util.Random.shuffle

class NeuralNetwork(sizes: List[Int]) {
	val layers = sizes.length
	val normal = breeze.stats.distributions.Gaussian(0, 1)
	var biases = for (y <- sizes.drop(1)) yield DenseMatrix.rand(y, 1, normal)
	var weights = for (t <- sizes.dropRight(1) zip sizes.drop(1)) yield DenseMatrix.rand(t._2, t._1, normal)

	def feedForward (activation: DenseMatrix[Double]) : DenseMatrix[Double] = {
		var output = activation

		biases zip weights foreach { case (bias, weight) => 
			output = sigmoid((weight * output) + bias)
		}

		return output
	}

	def sgd (trainingData: Seq[Tuple2[DenseMatrix[Double], DenseMatrix[Double]]], epochs: Int, miniBatchSize: Int, eta: Double, testData: Seq[Tuple2[DenseMatrix[Double], DenseMatrix[Double]]]) {
		val n = trainingData.length

		for (i <- 1 to epochs) {
			val data = shuffle(trainingData)
			val miniBatches = for (j <- 0 to n - 1 by miniBatchSize) yield trainingData.slice(j, j + miniBatchSize)

			miniBatches foreach { miniBatch => 
				updateMiniBatch(miniBatch, eta)
			}

			println(evaluate(testData) + "/" + testData.length + " correct")			
			println("Epoch " + i + " complete.")
		}
	}

	def updateMiniBatch (miniBatch: Seq[Tuple2[DenseMatrix[Double], DenseMatrix[Double]]], eta: Double) {
		var nabla_bias = for (bias <- biases) yield DenseMatrix.zeros[Double](bias.rows, bias.cols)
		var nabla_weight = for (weight <- weights) yield DenseMatrix.zeros[Double](weight.rows, weight.cols)

		miniBatch foreach { case (features, result) =>
			val (delta_nabla_bias, delta_nabla_weight) = backprop(features, result)
			nabla_bias = for (t <- nabla_bias zip delta_nabla_bias) yield t._1 + t._2
			nabla_weight = for (t <- nabla_weight zip delta_nabla_weight) yield t._1 + t._2	
		}

		weights = for (t <- weights zip nabla_weight) yield t._1 - (t._2 * (eta / miniBatch.length))
		biases = for (t <- biases zip nabla_bias) yield t._1 - (t._2 * (eta / miniBatch.length))
	}

	def backprop (features: DenseMatrix[Double], result: DenseMatrix[Double]) : (List[DenseMatrix[Double]], List[DenseMatrix[Double]]) = {
		var nabla_bias = for (bias <- biases) yield DenseMatrix.zeros[Double](bias.rows, bias.cols)
		var nabla_weight = for (weight <- weights) yield DenseMatrix.zeros[Double](weight.rows, weight.cols)
		
		var activation = features
		var activations = List(features)
		var zs: List[DenseMatrix[Double]] = List()

		biases zip weights foreach { case (bias, weight) => 
			val z = (weight * activation) + bias
			zs = zs :+ z
			activation = sigmoid(z)
			activations = activations :+ activation
		}

		var delta = (activations.reverse.head - result) :* sigmoid_prime(zs.reverse.head)
		nabla_bias = nabla_bias.updated(nabla_bias.length - 1, delta)
		nabla_weight = nabla_weight.updated(nabla_weight.length - 1, delta * activations.takeRight(2).head.t)

		for (i <- 2 to layers - 1) {
			val z = zs.takeRight(i).head
			val sp = sigmoid_prime(z)
			delta = (weights.takeRight(i - 1).head.t * delta) :* sp
			nabla_bias = nabla_bias.updated(nabla_bias.length - 1 - i, delta)
			nabla_weight = nabla_weight.updated(nabla_weight.length - 1 - i, delta * activations.takeRight(i + 1).head.t)
		}

		return (nabla_bias, nabla_weight)
	}

	def evaluate (test_data: Seq[Tuple2[DenseMatrix[Double], DenseMatrix[Double]]]) : Int = {
		val correct = for (t <- test_data if argmax(feedForward(t._1)) == argmax(t._2)) yield 1
		return correct.length
	}

	def sigmoid_prime (z: DenseMatrix[Double]) : DenseMatrix[Double] = {
		return sigmoid(z) :* (-sigmoid(z) + 1d)
	}
}

object NeuralNetwork {
	def apply(x: List[Int]) = new NeuralNetwork(x)
}