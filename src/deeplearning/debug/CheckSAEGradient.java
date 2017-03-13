package deeplearning.debug;

import org.jblas.DoubleMatrix;

import deeplearning.neuralnetwork.NeuralNetwork;
import deeplearning.sparseautoencoder.SparseAutoencoder;
/* Checks the implementation of Backpropagation Algorithm for Sparse Autoencoder
 * 
 * Uses Numerical Method
 * 
 *	Written By: Muhammad Zaheer
 *	Dated : 10th November, 2014
 * 
 * 
 * 
 * 
 * */
import deeplearning.util.CostGradient;
import deeplearning.util.MatrixUtil;

public class CheckSAEGradient {

	private int visibleSize;
	private int hiddenSize;
	private double lambda;
	private double sparsityParam;
	private double beta;
	private int m;

	private DoubleMatrix X;
	private DoubleMatrix nnParams;

	public CheckSAEGradient() {

		this.visibleSize = 6;
		this.hiddenSize = 3;
		this.m = 5;
		this.lambda = 0;
		this.sparsityParam = 0;
		this.beta = 0;
	}

	public CheckSAEGradient(int visibleSize, int hiddenSize,int m, double lambda,
			double sparsityParam, double beta) {

		this.visibleSize = visibleSize;
		this.hiddenSize = hiddenSize;
		this.sparsityParam = sparsityParam;
		this.beta = beta;
		this.lambda = lambda;
		this.m = m;
		
	}
	
public void check() {
		
		SparseAutoencoder sae = new SparseAutoencoder(this.visibleSize, this.hiddenSize,
			 this.lambda,this.sparsityParam,this.beta);
		
		sae.init();
		sae.setLambda(this.lambda);
		this.X = MatrixUtil.randInitializeWeights(m, this.visibleSize);

		sae.setX(this.X);
		sae.setM(this.m);
		this.nnParams = sae.getNnParams();
		
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
	
	CostGradient J_grad = SparseAutoencoder.nnCostFunction(nnParams, this.X,
			this.visibleSize, this.hiddenSize,
			this.m, this.lambda, this.sparsityParam, this.beta);

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

	public static void main(String[] args) {
		
		CheckSAEGradient c = new CheckSAEGradient(5,3,5,0,0.1,1);

		c.check();
		
	}

}
