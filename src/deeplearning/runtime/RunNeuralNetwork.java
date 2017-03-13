package deeplearning.runtime;

import deeplearning.neuralnetwork.*;
import deeplearning.util.CostGradient;

import org.jblas.*;
/*
 * RunNeuralNetwork right now is just a class which initializes and train a network 
 * using NeuralNetwork Class
 * This is where we set different knobs are experiment with our network
 * 
 * Written By: Muhammad Zaheer
 * Dated: 28-Sep-2014
 */


public class RunNeuralNetwork {
	

	public static void main(String[] args) {
		System.out.println("NeuralNetwork for MNIST Dataset");
		
		// Data file to be read
		String dataFile = "data/ex4data1.mat";
	
		//	hiddenLayerSize[] determines the no of hidden layers as well as no of neurons in each layer
		//	Below, we'er initializing two hidden layer neural network with 50 and 25 neurons each
		int hiddenLayerSize [] = new int [] {25};
		NeuralNetwork nn = new NeuralNetwork(400,hiddenLayerSize,10,0);
		//nn.init();
		nn.init(dataFile);
		
		//	Setting regularization parameter to zero
		nn.setLambda(0.001);
		
		//	Evaluating initial cost with randomly initialized data
		CostGradient J_grad = nn.nnCostFunction();
		
		System.out.println("Cost: " + J_grad.getCost());
		
		//	Training the network for 100 iterations
		nn.training(100);
		
		J_grad = nn.nnCostFunction();
		DoubleMatrix y = nn.getY();
		
		int [] p = nn.predict();
		
		
		System.out.println("Cost after training: " + J_grad.getCost());
		
		
	}

}
