/* NeuralNetwork.scala
*  
*  A class implementing a simple neural network and training it with mini batch stochastic gradient descent via
*  backpropagation. Inspired by Michael Nielsen's Python implementation in Neural Networks and Deep Learning http://neuralnetworksanddeeplearning.com/
*/

package network

import scala.util.Random.shuffle
import breeze.linalg._
import breeze.numerics._

class NeuralNetwork(sizes: Seq[Int]) {
	/* sizes contains the size of each layer of neurons in the network. So, if sizes were Seq(4, 3, 3, 2), the network would
	*  be a 4-layer network with 4 input neurons, two hidden layers of 3 neurons each, and an output layer with 2 neurons.
	*/
	val layers = sizes.length
	val normal = breeze.stats.distributions.Gaussian(0, 1)
	var biases = for (y <- sizes.drop(1)) yield DenseMatrix.rand(y, 1, normal)
	var weights = for (t <- sizes.dropRight(1) zip sizes.drop(1)) yield DenseMatrix.rand(t._2, t._1, normal)

	def feedForward (activation: DenseMatrix[Double]) : DenseMatrix[Double] = {
		/* Plug an activation into the network and return the output
		*/
		var output = activation

		biases zip weights foreach { case (bias, weight) => 
			output = sigmoid((weight * output) + bias)
		}

		return output
	}

	def sgd (trainingData: Seq[Tuple2[DenseMatrix[Double], DenseMatrix[Double]]], epochs: Int, miniBatchSize: Int, eta: Double, testData: Seq[Tuple2[DenseMatrix[Double], DenseMatrix[Double]]]) {
		/* Perform mini batch stochastic gradient descent to train the network, outputting the test accuracy at each epoch. The training
		*  and test data are both Seq[Tuple2[]] of DenseMatrix[Doubles], where each tuple is an input / label pair, and the rest
		*  of the arguments do what they say on the tin.
		*/
		val n = trainingData.length

		for (i <- 1 to epochs) {
			val data = shuffle(trainingData)
			val miniBatches = for (j <- 0 to n - 1 by miniBatchSize) yield trainingData.slice(j, j + miniBatchSize)

			miniBatches foreach { miniBatch => 
				updateMiniBatch(miniBatch, eta)
			}

			println(s"Epoch ${i} complete, with ${evaluate(testData)} / ${testData.length} correct")
		}
	}

	def updateMiniBatch (miniBatch: Seq[Tuple2[DenseMatrix[Double], DenseMatrix[Double]]], eta: Double) {
		/* Updates weights and biases via backpropagation over one minibatch. miniBatch is a Seq[Tuple2[]]
		*  of DenseMatrix[Double]s where each Tuple2 is an input / label pair, and eta
		*  is the learning rate.
		*/
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

	def backprop (features: DenseMatrix[Double], result: DenseMatrix[Double]) : (Seq[DenseMatrix[Double]], Seq[DenseMatrix[Double]]) = {
		/* Returns the gradient of the cost function as a Tuple2[] of DenseMatrix[Double]s, where nabla_bias
		*  and nabla_weight are both Seq[DenseMatrix[Double]] just like weights and biases
		*/
		var nabla_bias = for (bias <- biases) yield DenseMatrix.zeros[Double](bias.rows, bias.cols)
		var nabla_weight = for (weight <- weights) yield DenseMatrix.zeros[Double](weight.rows, weight.cols)
		
		// feedforward pass, storing z values
		var activation = features
		var activations = List(features)
		var zs: List[DenseMatrix[Double]] = List()

		biases zip weights foreach { case (bias, weight) => 
			val z = (weight * activation) + bias
			activation = sigmoid(z)
			zs = zs :+ z
			activations = activations :+ activation
		}

		// backward pass
		var delta = (activations.reverse.head - result) :* sigmoid_prime(zs.reverse.head)
		nabla_bias = nabla_bias.updated(nabla_bias.length - 1, delta)
		nabla_weight = nabla_weight.updated(nabla_weight.length - 1, delta * activations.takeRight(2).head.t)

		for (i <- 2 to layers - 1) {
			val z = zs.takeRight(i).head
			val sp = sigmoid_prime(z)
			delta = (weights.takeRight(i - 1).head.t * delta) :* sp
			nabla_bias = nabla_bias.updated(nabla_bias.length - i, delta)
			nabla_weight = nabla_weight.updated(nabla_weight.length - i, delta * activations.takeRight(i + 1).head.t)
		}

		return (nabla_bias, nabla_weight)
	}

	def evaluate (test_data: Seq[Tuple2[DenseMatrix[Double], DenseMatrix[Double]]]) : Int = {
		/* Returns the number of inputs from test_data for which the network's response is correct.
		*  The output is calculated as the index of the output neuron with the maximum activation.
		*/
		val correct = for (t <- test_data if argmax(feedForward(t._1)) == argmax(t._2)) yield 1
		return correct.length
	}

	def sigmoid_prime (z: DenseMatrix[Double]) = sigmoid(z) :* (-sigmoid(z) + 1.0)
}

object NeuralNetwork {
	def apply(x: Seq[Int]) = new NeuralNetwork(x)
}