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
	type Datum = Tuple2[DenseMatrix[Double], DenseMatrix[Double]]

	val layers = sizes.length
	val normal = breeze.stats.distributions.Gaussian(0, 1)
	var biases = for (y <- sizes.drop(1)) yield DenseMatrix.rand(y, 1, normal)
	var weights = for ((x, y) <- sizes.dropRight(1) zip sizes.drop(1)) yield DenseMatrix.rand(y, x, normal)

	// Derivative of sigmoid function
	def sigmoid_prime (z: DenseMatrix[Double]) = sigmoid(z) :* (-sigmoid(z) + 1.0)

	// Plug an activation into the network and return the output
	def feedForward (activation: DenseMatrix[Double]) : DenseMatrix[Double] = {
		var output = activation

		biases zip weights foreach { case (bias, weight) => 
			output = sigmoid((weight * output) + bias)
		}

		return output
	}

	/* Perform mini batch stochastic gradient descent to train the network, outputting the test accuracy at each epoch. The training
	*  and Optional test data are both Seq[Tuple2[]] of DenseMatrix[Doubles], where each tuple is an input / label pair, and the rest
	*  of the arguments do what they say on the tin. If we provide testData, we get an evaluation on our test set printed for each epoch
	*/
	def sgd (trainingData: Seq[Datum], epochs: Int, miniBatchSize: Int, eta: Double, testData: Option[Seq[Datum]]) {
		val n = trainingData.length

		for (i <- 1 to epochs) {
			val data = shuffle(trainingData)
			val miniBatches = for (j <- 0 to n - 1 by miniBatchSize) yield trainingData.slice(j, j + miniBatchSize)

			miniBatches foreach { miniBatch => 
				updateMiniBatch(miniBatch, eta)
			}

			testData match {
				case Some(data) => println(s"Epoch ${i} complete, with ${evaluate(data)} / ${data.length} correct")
				case None => println(s"Epoch ${i} complete")
			}
		}
	}

	/* Updates weights and biases via backpropagation over one minibatch. miniBatch is a Seq[Tuple2[]]
	*  of DenseMatrix[Double]s where each Tuple2 is an input / label pair, and eta
	*  is the learning rate.
	*/
	def updateMiniBatch (miniBatch: Seq[Datum], eta: Double) {
		var nabla_bias = for (bias <- biases) yield DenseMatrix.zeros[Double](bias.rows, bias.cols)
		var nabla_weight = for (weight <- weights) yield DenseMatrix.zeros[Double](weight.rows, weight.cols)

		miniBatch foreach { case (features, result) =>
			val (delta_nabla_bias, delta_nabla_weight) = backprop(features, result)
			nabla_bias = for ((nabla, delta) <- nabla_bias zip delta_nabla_bias) yield nabla + delta
			nabla_weight = for ((nabla, delta) <- nabla_weight zip delta_nabla_weight) yield nabla + delta
		}

		weights = for ((weight, nabla) <- weights zip nabla_weight) yield weight - (nabla * (eta / miniBatch.length))
		biases = for ((bias, nabla) <- biases zip nabla_bias) yield bias - (nabla * (eta / miniBatch.length))
	}

	/* Returns the gradient of the cost function as a Tuple2[] of DenseMatrix[Double]s, where nabla_bias
	*  and nabla_weight are both Seq[DenseMatrix[Double]] just like weights and biases
	*/
	def backprop (features: DenseMatrix[Double], result: DenseMatrix[Double]) : (Seq[DenseMatrix[Double]], Seq[DenseMatrix[Double]]) = {
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

		for (i <- 2 until layers) {
			val z = zs.takeRight(i).head
			val sp = sigmoid_prime(z)
			delta = (weights.takeRight(i - 1).head.t * delta) :* sp
			nabla_bias = nabla_bias.updated(nabla_bias.length - i, delta)
			nabla_weight = nabla_weight.updated(nabla_weight.length - i, delta * activations.takeRight(i + 1).head.t)
		}

		return (nabla_bias, nabla_weight)
	}

	/* Returns the number of inputs from test_data for which the network's response is correct.
	*  The output is calculated as the index of the output neuron with the maximum activation.
	*/
	def evaluate (test_data: Seq[Datum]) = (for ((input, label) <- test_data if argmax(feedForward(input)) == argmax(label)) yield 1).length
}

object NeuralNetwork {
	def apply(x: Seq[Int]) = new NeuralNetwork(x)
}