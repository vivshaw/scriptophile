package scriptophile

import breeze.linalg._
import breeze.numerics._

import network.NeuralNetwork

object Mycophile extends App {
	var net = NeuralNetwork(List(2, 3, 2))

	val zero = new DenseMatrix(2,1,DenseVector(0.0, 1.0).toArray)
	val one = new DenseMatrix(2,1,DenseVector(1.0, 0.0).toArray)
	val both = new DenseMatrix(2,1,DenseVector(1.0, 1.0).toArray)
	val neither = new DenseMatrix(2,1,DenseVector(0.0, 0.0).toArray)

	val excl_or = List((zero, one), (one, one), (both, zero), (neither, zero))
	val incl_or = List((zero, one), (one, one), (both, one), (neither, zero))
	val train_data = excl_or
	
	val bufferedSource = io.Source.fromFile("src/main/resources/mnist_train.csv")
    for (line <- bufferedSource.getLines) {
        val cols = line.split(",").map(_.trim)
        // do whatever you want with the columns here
        println(s"${cols(0)}")
    }
    bufferedSource.close

	/*
	println("now training")
	net.sgd(train_data, 10000, 4, 0.5, train_data)
	*/
}