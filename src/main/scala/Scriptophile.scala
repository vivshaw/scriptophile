package scriptophile

import scala.io.Source

import breeze.linalg._
import breeze.numerics._

import network.NeuralNetwork

object Scriptophile extends App {	
    def digitToArray (digit: Int) : DenseMatrix[Double] = {
    	val vect = DenseVector.zeros[Double](10)
    	vect(digit) = 1.0
    	var digitMatrix = new DenseMatrix(10, 1, vect.toArray)
    	return digitMatrix
    }

    def pixelToArray (pixel: Array[Double]) : DenseMatrix[Double] = {
    	val out = new DenseMatrix(784, 1, pixel)
    	return out
    }

	case class mnistDatum(line: String) {
		val raw = line.split(",").map(_.trim)
		val data = (pixelToArray(raw.tail map(item => item.toDouble / 255.0)),
					digitToArray(raw.head.toInt))
	}
	
    val mnistRaw = Source.fromFile("src/main/resources/mnist_train.csv") getLines() drop(1) map(line => mnistDatum(line))
    val mnistData = mnistRaw map(datum => datum.data)
    val mnist = mnistData.toSeq

    val mnist_train = mnist.dropRight(10000)
    val mnist_test = mnist.takeRight(10000)
    
	println("now training")
	val net = NeuralNetwork(List(784, 30, 10))
	net.sgd(mnist_train, 30, 10, 3.0, mnist_test)

	println(s"final accuracy: ${ net.evaluate(mnist_test).toDouble / mnist_test.length.toDouble }%")
}