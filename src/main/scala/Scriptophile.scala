package scriptophile

import scala.io.Source
import breeze.linalg._
import breeze.numerics._

import network.NeuralNetwork

object Scriptophile extends App {	
    // Turns an Int digit into a (10,1) DenseMatrix one-hot encoded representation of the label
    def labelToMatrix (label: Int) : DenseMatrix[Double] = {
    	var onehot = DenseVector.zeros[Double](10)
    	onehot(label) = 1.0
    	return new DenseMatrix(10, 1, onehot.toArray)
    }

    // Turns an array of doubles into a (784,1) DenseMatrix representing the 784 pixels of an image from the MNIST data.
    def imageToMatrix (image: Array[Double]) = new DenseMatrix(784, 1, image)

    /* Case class used to process the raw CSV data into an Array[Double] containing the image's pixel data,
    *  and an Int label. Also performs feature rescaling on the image's pixel values to [0, 1].
    */
	case class mnistDatum(line: String) {
		val raw   = line.split(",") map(_.trim)
        val label = labelToMatrix(raw.head.toInt)
        val image = imageToMatrix(raw.tail map(item => item.toDouble / 255.0))
	}
	
    /* Load MNIST data from a 785-column CSV where column 0 is the label (a digit from 0 to 9), and columns 1...784
    *  each represent one pixel from the 28x28 MNIST image. These both get transformed into (n,1) matrices, then put
    *  into a List[Tuple2[]] where the first element of each tuple is an image, and the second element is its label.
    */
    val mnist = (Source.fromFile("src/main/resources/mnist_train.csv")
                       .getLines()
                       .drop(1)
                       .map { line => mnistDatum(line) }
                       .map { datum => (datum.image, datum.label) }).toSeq

    // Test/train split
    val mnist_train = mnist.dropRight(10000)
    val mnist_test = mnist.takeRight(10000)
    
    /* Initialize 3-layer neural network with 784 input neurons corresponding to te pixel of a MNIST image, a 30-neuron hidden layer,
    *  and 10 output neurons corresponding to the 10 digit labels.
    */
    val net = NeuralNetwork(List(784, 30, 10))

	println("now training")
    // Train for 30 epochs, with mini-batch size 10, and learning rate 3.0
	net.sgd(mnist_train, 30, 10, 3.0, Some(mnist_test))

	println(s"final accuracy: ${ 100.0 * net.evaluate(mnist_test).toDouble / mnist_test.length.toDouble }%")
}