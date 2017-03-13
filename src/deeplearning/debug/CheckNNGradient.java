package deeplearning.debug;

import org.jblas.DoubleMatrix;
import deeplearning.neuralnetwork.*;
import deeplearning.util.CostGradient;
import deeplearning.util.MatrixUtil;

/* Checks the implementation of Backpropagation Algorithm for Neural Network
 * 
 * Uses Numerical Method
 * 
 *	Written By: Muhammad Zaheer
 *	Dated : 25th October, 2014
 * 
 * 
 * 
 * 
 * */

public class CheckNNGradient {
	private int inputLayerSize;
	private int hiddenLayerSize[];
	private int numLabels;
	private double lambda;
	private int m;

	private DoubleMatrix X;
	private DoubleMatrix y;
	private DoubleMatrix nnParams;

	public CheckNNGradient() {

		this.inputLayerSize = 3;
		this.hiddenLayerSize = new int []{6};
		this.numLabels = 3;
		this.m = 5;
		this.lambda = 0;

	}

	public CheckNNGradient(int inputLayerSize, int hiddenLayerSize[],
			int numLabels,double lambda,int m) {

		this.inputLayerSize = inputLayerSize;
		this.hiddenLayerSize = hiddenLayerSize;
		this.numLabels = numLabels;
		this.m = m;
		this.lambda = lambda;

	}

	public void check() {
		
		NeuralNetwork nn = new NeuralNetwork(this.inputLayerSize, this.hiddenLayerSize,
				this.numLabels, this.lambda);
		
		nn.init();
		nn.setLambda(this.lambda);
		this.X = MatrixUtil.randInitializeWeights(this.inputLayerSize, m);
		this.y = DoubleMatrix.zeros(m, 1);
		for (int i = 0; i < y.rows; i++) {
			y.put(i, 1 + (double) i % this.numLabels);
		}

		nn.setX(this.X);
		nn.setY(this.y);
		nn.setM(this.m);
		this.nnParams = nn.getNnParams();
		
		CostGradient J_grad  = J(nnParams);
		
		DoubleMatrix realGrad = J_grad.getGradient();
		DoubleMatrix numGrad = computeNumericalGradients(this.nnParams);
		//System.out.println("READ GRAD");
		//MatrixUtil.printMatrix(realGrad);
		double sum= 0.0;
		for (int i =0; i<numGrad.length;i++) {
			
			System.out.println (realGrad.get(i) + " " + numGrad.get(i));
			sum = sum + Math.abs(realGrad.get(i) - numGrad.get(i));
			}
		
		System.out.println ("Norm diff: " +sum);
	}
	
	public CostGradient J (DoubleMatrix nnParams) {
		
		CostGradient J_grad = NeuralNetwork.nnCostFunction(nnParams, this.X,
				this.y, this.inputLayerSize, this.hiddenLayerSize, this.numLabels,
				this.m, this.lambda);

		return J_grad;
		
		
	}
	
	public DoubleMatrix computeNumericalGradients(DoubleMatrix theta) {
		
		DoubleMatrix numGrad = DoubleMatrix.zeros (theta.rows,theta.columns);
		DoubleMatrix perturb = DoubleMatrix.zeros(theta.rows,theta.columns);
		
		double e = 0.0001;
		
		for (int p = 0; p < (theta.rows*theta.columns); p++) {
			perturb.put(p, e);
			CostGradient J1 = J(theta.sub(perturb));
			CostGradient J2 = J(theta.add(perturb));
			
			double temp = (J2.getCost() - J1.getCost())/ (2*e); 
			numGrad.put(p,temp);
			perturb.put(p, 0);
		}
		
		
		
		return numGrad;
	}


	public static void main(String args[]) {
		
		
		int hiddenLayerSize [] = new int[] {3};
		CheckNNGradient c = new CheckNNGradient(5,hiddenLayerSize,5,0.003,10);

		c.check();

	}

}
