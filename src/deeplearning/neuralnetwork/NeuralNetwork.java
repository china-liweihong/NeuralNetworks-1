package deeplearning.neuralnetwork;

import org.jblas.DoubleMatrix;
import deeplearning.io.*;
import deeplearning.minimizers.*;
import deeplearning.util.Activations;
import deeplearning.util.CostFunctions;
import deeplearning.util.CostGradient;
import deeplearning.util.MatrixUtil;
import deeplearning.util.Parameters;

/*
 * NeuralNetwork Class provides the functionality to implement a 
 * fully connected simple neural network with configurable no of layers 
 * and neurons in each layer
 * 
 * Written By: Muhammad Zaheer
 * Dated: 03-Oct-2014
 * 
 */

public class NeuralNetwork implements deeplearning.minimizers.Network {

	private int inputLayerSize;
	private int hiddenLayerSize[];	//	There could be multiple hidden layers
	private int numLabels;
	private double lambda;					//	Regularization parameters
	private int m;									//	No of training examples

	private DoubleMatrix X;					//	Input Matrix M x N 
	private DoubleMatrix y;					//	Input labels M x 1
	private DoubleMatrix W[];				//	Parameters for Neurons
	private DoubleMatrix b[];				//	Parameters for bias unit 
	private DoubleMatrix nnParams;	//	A Vector of all the parameters i.e. rolled version of W and b

	
	public NeuralNetwork() {
	}

	public NeuralNetwork(int inputLayerSize, int hiddenLayerSize[],
		int numLabels, double lambda) {
		this.inputLayerSize = inputLayerSize;
		this.hiddenLayerSize = hiddenLayerSize;
		this.numLabels = numLabels;
		this.lambda = lambda;
	}
	
	// Randomly initializes parameters into W[] and b[] and rolls them into nnParams
	// Size of matrices depends upon inputLayerSize, numLabels and hiddenLayerSize[]
	public void initParameters() {
		
		this.W = new DoubleMatrix [hiddenLayerSize.length + 1];
		this.b = new DoubleMatrix [hiddenLayerSize.length + 1];
		
		for (int i = 0; i < W.length; i++) {
			if (i == 0) {
				this.W[i] = MatrixUtil.randInitializeWeights(this.inputLayerSize,this.hiddenLayerSize[i]);
				this.b[i] = MatrixUtil.randInitializeWeights(1, this.hiddenLayerSize[i]);
			}
			else if (i == (W.length - 1)) {
				this.W[i] = MatrixUtil.randInitializeWeights(this.hiddenLayerSize[i-1], this.numLabels);
				this.b[i] = MatrixUtil.randInitializeWeights(1,this.numLabels);
			}
			else {
				this.W[i] = MatrixUtil.randInitializeWeights(this.hiddenLayerSize[i-1],this.hiddenLayerSize[i]);
				this.b[i] = MatrixUtil.randInitializeWeights(1,this.hiddenLayerSize[i]);
			}
		}
		
		this.nnParams = MatrixUtil.rollParameters(W,b);
	}
	
	// Regular init just initializes parameters
	public void init() {
		initParameters();
	}	

	// Loads data into X and y, randomly initializes parameters into W[] and b[]
	// and rolls them into nnParams
	public void init(String dataFile) {

		this.loadData(dataFile);
		this.initParameters();
	}

	// Loads data into X and y, and parameters into Theta[] and nn_params
	// from the specified files
	public void init(String dataFile, String parameterFile, int count) {

		this.loadData(dataFile);

		//TODO Correct loadParameters for seperate W and b
		this.loadParameters(parameterFile, hiddenLayerSize.length + 1);
		this.nnParams = MatrixUtil.rollParameters(this.W,this.b);
	}

	// loads data from a given .mat file into X and y using Loader class
	public void loadData(String dataFile) {
		// count = 2 because we want to read two matrices from the mat file i.e. X
		// and y
		int count = 2;
		DoubleMatrix[] matrices = Loader.loadMatrix(dataFile, count);

		this.X = matrices[0];
		this.y = matrices[1];

		System.out.println("Data loaded in X and y");

		this.m = X.rows;
	}

	// TODO: Deprecated function, Never use it
	// First tailor it for W[] and b[]
	public void loadParameters(String parameterFile, int count) {

		DoubleMatrix[] matrices = Loader.loadMatrix(parameterFile, count);

		System.out.println("Parameters loaded in Theta []");
	}

	// Sets regularization parameter lambda
	public void setLambda(double lambda) {
		
		this.lambda = lambda;
		return;

	}

	public CostGradient nnCostFunction() {

		CostGradient J_grad = nnCostFunction(this.nnParams, this.X,
				this.y, this.inputLayerSize, this.hiddenLayerSize, this.numLabels,
				this.m, this.lambda);

		return J_grad;

	}
	
	// This interface is useful for Diagnosis when we increment our training set
	// Variance vs Bias test
	public CostGradient nnCostFunction(DoubleMatrix X,
			DoubleMatrix y) {

		CostGradient J_grad = nnCostFunction(this.nnParams, X, y,
				this.inputLayerSize, this.hiddenLayerSize, this.numLabels, X.rows,
				this.lambda);
		return J_grad;
	}

	// Calculates cost and gradient w.r.t a certain set of parameters
	// This function would be used by a minimization function such as fminunc
	public static CostGradient nnCostFunction(DoubleMatrix nnParams, DoubleMatrix X, DoubleMatrix y, int inputLayerSize,
			int hiddenLayerSize[], int numLabels, int m, double lambda) {

		// Variable J to hold cost
		double J = 0;
		
		// grad matrix would hold gradient for each parameter
		DoubleMatrix grad = DoubleMatrix.zeros(nnParams.rows, nnParams.columns);
		
		// Get W and b from nnParams
		Parameters p = reshape(nnParams,inputLayerSize, numLabels,hiddenLayerSize);
		DoubleMatrix W[] = p.getW();
		DoubleMatrix b[] = p.getB();
		
		// W_grad[] and b_grad[] are gradient matrices for reshaped
		// parameters i.e. W[] and b[]
		DoubleMatrix W_grad []  = new DoubleMatrix[W.length];
		DoubleMatrix b_grad [] = new DoubleMatrix[b.length];
		
		// Initializing gradients to zeros
		for (int i = 0; i < W.length; i++) {
			W_grad[i] = DoubleMatrix.zeros(W[i].rows,W[i].columns);
			b_grad[i] = DoubleMatrix.zeros(b[i].rows,b[i].columns);
			
		}
		
		// Y is the matrix to hold the translated label (0, 1, 2, --- , 9)
		// Y is a matrix where i'th row represents the i'th training example
		// In i'th training example:
		// Index corresponding to the label 'k' would be '1', rest would be zero
		
		DoubleMatrix Y = DoubleMatrix.zeros(m, numLabels);
		// Tranlating labels into vectors || Handling the corner case for 10 as well
		for (int i = 0; i < m; i++) {
			if ((int) y.get(i) != numLabels)
				Y.put(i, (int) y.get(i), 1);
			else
				Y.put(i, 0, 1);
		}
		
		// a[] -> Activations
		DoubleMatrix a[] = new DoubleMatrix [W.length+1];
		
		// z[] -> Weighted Sums
		DoubleMatrix z[]  = new DoubleMatrix [W.length+1];
		
		for (int k = 0; k < a.length; k++) {
			// k == 0 i.e. input layer units are the first activation units
			if (k == 0) {
				// Extracting i'th training example
				a[k] = X.transpose(); 
			}
			else {
					
				// Getting the weighted sum i.e. z = W.a + b
				
				z[k] = W[k-1].mmul(a[k-1]).addColumnVector(b[k-1]); 
				// Getting the activation by sigmoid function i.e. a = sigmoid(z)
				a[k] = Activations.sigmoid(z[k]);
			}
		}
		
		// Final activations are our hypothesis h
		// Dimensions of h : 10 x 5000
		DoubleMatrix hn = a[a.length-1];
		
		
		// Dimensions of Y : 5000 x 10
		J = CostFunctions.logarithmic(Y, hn.transpose())/m;
		// Regularizing cost
		J = J + ((lambda / 2) * MatrixUtil.squareSum(W));

		DoubleMatrix sn[] = new DoubleMatrix[a.length];
		sn[sn.length - 1] = hn.sub(Y.transpose());
		
		for (int k = sn.length - 2; k > 0; k--) {
			sn[k] = (W[k].transpose().mmul(sn[k+1])).mul(Activations.sigmoidGradient(z[k]));
		}
		for (int k = 0 ; k<W.length; k++) {
			W_grad[k] = sn[k+1].mmul(a[k].transpose());
			b_grad[k] = sn[k+1].rowSums();
		}
		

		for (int k = 0; k < W.length; k++) {
			W_grad[k] = W_grad[k].div(m);
			b_grad[k] = b_grad[k].div(m);
		}

		// Regularizing gradients
		for (int k = 0; k < W.length; k++) {
				W_grad[k] = W_grad[k].add(W[k].mul(lambda));
		}

		// Rolling gradients into a single vector
		grad = MatrixUtil.rollParameters(W_grad,b_grad);
		
		CostGradient J_gradn = new CostGradient(J,grad);
		// Creating an object to return cost along with gradient
		return J_gradn;

	}
	
	// Trains the network for 'iter' no of iterations using backpropagation algorithm
	public void training(int iter) {
		FminuncBatch.minimize(iter, this);
	}
	
	// A simple batch gradient descent || Works slowly but is simple
	public void gradientDescent(int i) {

		double alpha = 0.2;
		CostGradient cg = this.nnCostFunction();

		nnParams = nnParams.sub(cg.getGradient().mul(alpha));
		System.out.println("Cost " + i + ": " + cg.getCost());

	}

	// Used for calculating accuracy and providing prediction on the basis of trained model
	public int[] predict() {
		
		Parameters p  = reshape(this.nnParams, this.inputLayerSize,this.numLabels,this.hiddenLayerSize);
		
		return predict(p.getW(),p.getB(), this.X);

	}
	
	public int[] predict(DoubleMatrix W[],DoubleMatrix b[], DoubleMatrix X) {
		
		DoubleMatrix h = X.mmul(W[0].transpose()).addRowVector(b[0]);
		h = Activations.sigmoid(h);
		
		for (int i = 1; i < W.length; i++) {
			h = h.mmul(W[i].transpose()).addiRowVector(b[i]);
			h = Activations.sigmoid(h);
		}

		int[] p = h.rowArgmaxs();
		// Calculating training accuracy
		double t = 0;
		
		for (int i = 0; i < m; i++) {
			int yi = (int) y.get(i, 0);

			if (yi == 10)
				yi = 0;
			if (p[i] == yi)
				t = t + 1;
		}
		
		t = t / this.m;

		t = t * 100;

		System.out.println("Training accruacy: " + t);
		
		return p;
	}
	
	// Reshapes nnParams into W[] and b[] and return them as Parameters object
	public static Parameters reshape(DoubleMatrix nnParams, int inputLayerSize, int numLabels, int hiddenLayerSize[]) {
		
	
		DoubleMatrix W [] = new DoubleMatrix[hiddenLayerSize.length + 1];
		DoubleMatrix b[] = new DoubleMatrix [hiddenLayerSize.length + 1];
		
		int start = 0, end = 0;
		int endb  = 0;
		for (int i = 0; i < W.length; i++) {
			if (i == 0) {
				start = 0;
				endb = hiddenLayerSize[i];
				end = hiddenLayerSize[i] * (1 + inputLayerSize);
				
				b[i] = new DoubleMatrix(hiddenLayerSize[i],1,nnParams.getRange(start,endb).toArray());
				W[i] = new DoubleMatrix(hiddenLayerSize[i],inputLayerSize,nnParams.getRange(endb,end).toArray());
				
			} else if (i == (W.length - 1)) {
				start = end;
				endb = start + numLabels;
				end = nnParams.rows;
				b[i] = new DoubleMatrix(numLabels,1,nnParams.getRange(start, endb).toArray());
				W[i] = new DoubleMatrix(numLabels, hiddenLayerSize[i-1], nnParams.getRange(endb,end).toArray());
			} else {
				start = end;
				endb = start + (hiddenLayerSize[i]);
				end = start + (hiddenLayerSize[i] * (1 + hiddenLayerSize[i - 1]));
				b[i] = new DoubleMatrix(hiddenLayerSize[i],1, nnParams.getRange(start, endb).toArray());
				W[i] = new DoubleMatrix(hiddenLayerSize[i],hiddenLayerSize[i-1],nnParams.getRange(endb,end).toArray());

			}
			
		}
		return new Parameters(W,b);
	}

	// GETTERS AND SETTERS
	public DoubleMatrix getX() {
		return X;
	}

	public void setX(DoubleMatrix x) {
		X = x;
		this.m = x.rows;
	}

	public DoubleMatrix getY() {
		return y;
	}

	public void setY(DoubleMatrix y) {
		this.y = y;
		this.m = y.rows;
	}

	public int[] getHiddenLayerSize() {
		return hiddenLayerSize;
	}

	public void setHiddenLayerSize(int[] hiddenLayerSize) {
		this.hiddenLayerSize = hiddenLayerSize;
	}

	public DoubleMatrix getNnParams() {
		return nnParams;
	}

	public void setNnParams(DoubleMatrix nnParamsNew) {
		this.nnParams = nnParamsNew;
	}

	public void setM(int m) {
		this.m = m;

	}
	
	

}
