package deeplearning.sparseautoencoder;

import org.jblas.*;

import deeplearning.io.*;
import deeplearning.minimizers.FminuncBatch;
import deeplearning.util.Activations;
import deeplearning.util.CostFunctions;
import deeplearning.util.CostGradient;
import deeplearning.util.MatrixUtil;
import deeplearning.util.Parameters;

import org.jblas.ranges.IntervalRange;

/* SparseAutoencoder Class
 * 
 * Written By: Muhammad Zaheer
 * Date Started: 28 OCT, 2014
 */

public class SparseAutoencoderLinearCost implements deeplearning.minimizers.Network {
	//8*8 = 64 input pixels
	private int visibleSize;
	private int hiddenSize;
	private double sparsityParam;

	private double lambda;
	private double beta;
	private int m;
	private DoubleMatrix X;
	private DoubleMatrix nnParams;

	// w[0] and b[0] to encode, and w[1] and b[1] to decode
	DoubleMatrix W[];
	DoubleMatrix b[];

	// TODO: Decouple images & patches from here.
	DoubleMatrix images[];
	DoubleMatrix patches[];

	public SparseAutoencoderLinearCost() {
		
		this.visibleSize = 8 * 8;
		this.hiddenSize = 25;
		this.sparsityParam = 0.01;
		this.lambda = 0.0001;
		this.beta = 3;

	}

	public SparseAutoencoderLinearCost(int visibleSize, int hiddenSize, double lambda,
			double sparsityParam, double beta) {

		this.visibleSize = visibleSize;
		this.hiddenSize = hiddenSize;
		this.sparsityParam = sparsityParam;
		this.lambda = lambda;
		this.beta = beta;

	}

	public void init() {
		this.initParameters();
	}

	public void init(String dataFile) {

		this.loadData(dataFile);

		// TODO : Initialize Parameters

	}

	public void loadData(String dataFile) {
		/*
		 * We have to read only 1-matrix i.e. IMAGES IMAGES is of dimensions 512 *
		 * 512 * 10 i.e. 10 images of 512*512
		 * 
		 * JBLAS is limited to only 2 dimensional matrices i.e. we get a matrix of
		 * dimensions 512*5120 Therefore, we loop through this matrix to restore the
		 * images in an array of DoubleMatrices
		 */

		int count = 1;
		DoubleMatrix[] matrices = Loader.loadMatrix(dataFile, count);

		DoubleMatrix IMAGES = matrices[0];

		DoubleMatrix[] images = new DoubleMatrix[10];

		IntervalRange rowsRange, colsRange;

		for (int i = 0; i < 10; i++) {
			images[i] = DoubleMatrix.zeros(512, 512);
			rowsRange = new IntervalRange(0, 512);
			colsRange = new IntervalRange(i * 512, (i + 1) * 512);

			images[i] = IMAGES.get(rowsRange, colsRange);

		}

		this.images = images;

	}

	// TODO: Very terrible code, redo it later
	public void generateTrainingSet(int numOfPatches, int patchSize) {
		java.util.Random rand = new java.util.Random();
		DoubleMatrix[] patches = new DoubleMatrix[numOfPatches];

		// DoubleMatrix[] matrix = this.images;
		/*
		 * preparing 8*8 patches from randomly selected 10 images... and using an
		 * random X and Y coordinate to select 8*8 patch from... the 512*512 matrix.
		 */
		for (int k = 0; k < numOfPatches; k++) {
			// generating random points
			int randomY = rand.nextInt(512 - patchSize);
			int randomX = rand.nextInt(512 - patchSize);
			int RandomImage = rand.nextInt(10);
			// initializing 10000 patches to zeros of 8*8
			patches[k] = DoubleMatrix.zeros(patchSize, patchSize);
			// putting matrix[i+randX,j+randY] into patches[k]

			// TODO: Vectorize this loop using JBlas's interval Range
			for (int i = 0; i < patchSize; i++) {
				for (int j = 0; j < patchSize; j++) {
					patches[k].put(i, j,
							images[RandomImage].get(randomX + i, randomY + j));
				}
			}
		}

		// matrix with means of columns for all patches
		DoubleMatrix[] meanMatrix = new DoubleMatrix[numOfPatches];
		// matrix with only means of all patches
		DoubleMatrix meanVector = DoubleMatrix.zeros(numOfPatches);
		DoubleMatrix[] temp = new DoubleMatrix[numOfPatches];

		double[] sumofarray = new double[numOfPatches];
		for (int i = 0; i < numOfPatches; i++) {
			meanMatrix[i] = DoubleMatrix.zeros(patchSize);
			// getting column from patches, and put their mean in meanMatrix[10000]
			for (int j = 0; j < patchSize; j++) {
				// System.out.println("Getting column"+patches[i].getColumn(j));
				meanMatrix[i].put(j, patches[i].getColumn(j).mean());
			}
			// System.out.println("Getting column means" + meanMatrix[0]);
			// putting the means of meanMatrix to meanVectors
			meanVector.put(i, meanMatrix[i].mean());
			// System.out.println("mean from meanVector" + meanVector.get(0));
			// subtracting mean from patches element-wise, temp[i] is 8*8
			temp[i] = patches[i].sub(meanVector.get(i));
			// System.out.println("Getting the elements of patch before sub"+
			// patches[0]);
			// System.out.println("Getting the elements of patch after sub"+ temp[0]);
		}
		// standard deviation of all the packages
		DoubleMatrix unrolledpatch = deeplearning.util.MatrixUtil
				.rollParameters(patches);
		DoubleMatrix unrolled = DoubleMatrix.zeros(numOfPatches * patchSize
				* patchSize);
		double unrolledmean = unrolledpatch.mean();
		unrolled = unrolledpatch.sub(unrolledmean);
		DoubleMatrix squarederror = unrolled.mul(unrolled);
		double sumsqrderror = squarederror.sum();
		sumsqrderror = sumsqrderror / (numOfPatches * patchSize * patchSize);
		double std = Math.sqrt(sumsqrderror);
		for (int i = 0; i < numOfPatches; i++) {
			double pstd = 3 * (std);
			// System.out.println("\n patche value "+patches[0]);
			// System.out.println("\n patch value after multiplying by std"+pstd);
			DoubleMatrix min = patches[i].min(pstd);
			// System.out.println("\n finding min from pstd and patches"+min);
			DoubleMatrix max = min.max(-1 * pstd);
			// System.out.println("\n finding max from -pstd and min(patches,pstd)"+max);
			max = max.div(pstd);
			// System.out.println("\n patche after dividing by pstd"+patches[0]);
			max = max.add(1);
			// System.out.println("\nafter adding 1"+patches[0]);
			max = max.mul(0.4);
			// System.out.println("\n after multipluing by .4"+patches[0]);
			patches[i] = max.add(0.1);
			// System.out.println("\n after adding 0.1"+patches[0]);
		}
		// unrolling patches
		// DoubleMatrix [] temppatches = patches;
		/*
		 * for(int i=0; i<temppatches.length; i++) { temppatches[i] =
		 * temppatches[i].reshape(1, temppatches[i].rows*temppatches[i].columns); }
		 */
		this.X = deeplearning.util.MatrixUtil.rollParameters(patches)
				.reshape(patchSize * patchSize, numOfPatches);
		// System.out.println(patches[rand.nextInt(numofpatches)]);
		this.patches = patches;

		// super.setX(patches[0]);
		this.m = numOfPatches;
	}
	
	// Initializes encoding and decoding parameters
	public void initParameters() {

		W = new DoubleMatrix[2];
		b = new DoubleMatrix[2];
		// W[0] : 64 x 25
		// b[0] : 1 x 25
		W[0] = MatrixUtil.randInitializeWeights(this.visibleSize, this.hiddenSize);
		b[0] = MatrixUtil.randInitializeWeights(1, this.hiddenSize);

		// W[1] : 25 x 64
		// b[1] : 1 x 64
		W[1] = MatrixUtil.randInitializeWeights(this.hiddenSize, this.visibleSize);
		b[1] = MatrixUtil.randInitializeWeights(1, this.visibleSize);
		this.nnParams = MatrixUtil.rollParameters(W, b);

	}

	// Sets regularization parameter lambda
	public void setLambda(double lambda) {

		this.lambda = lambda;
		return;

	}

	public CostGradient nnCostFunction() {

		CostGradient J_grad = nnCostFunction(this.nnParams, this.X,
				this.visibleSize, this.hiddenSize, this.m, this.lambda,
				this.sparsityParam, this.beta);

		return J_grad;

	}

	// This interface is useful for Diagnosis when we increment our training set
	// Variance vs Bias test
	public CostGradient nnCostFunction(DoubleMatrix X, DoubleMatrix y) {

		CostGradient J_grad = nnCostFunction(this.nnParams, X, this.visibleSize,
				this.hiddenSize, this.X.rows, this.lambda, this.sparsityParam,
				this.beta);
		return J_grad;
	}

	// Calculates cost and gradient w.r.t a certain set of parameters
	// This function would be used by a minimization function such as fminunc
	public static CostGradient nnCostFunction(DoubleMatrix nnParams,
			DoubleMatrix X, int visibleSize, int hiddenSize, int m, double lambda,
			double sparsityParam, double beta) {
		// Variable J to hold cost
		double J = 0;

		// grad matrix would hold gradient for each parameter
		DoubleMatrix grad = DoubleMatrix.zeros(nnParams.rows, nnParams.columns);

		// Get W and b from nnParams
		Parameters p = reshape(nnParams, visibleSize, hiddenSize);
		DoubleMatrix W[] = p.getW();
		DoubleMatrix b[] = p.getB();
		
		// W_grad[] and b_grad[] are gradient matrices for reshaped
		// parameters i.e. W[] and b[]
		DoubleMatrix W_grad[] = new DoubleMatrix[W.length];
		DoubleMatrix b_grad[] = new DoubleMatrix[b.length];

		// Initializing gradients to zeros
		for (int i = 0; i < W.length; i++) {
			W_grad[i] = DoubleMatrix.zeros(W[i].rows, W[i].columns);
			b_grad[i] = DoubleMatrix.zeros(b[i].rows, b[i].columns);

		}

		// a[] -> Activations
		DoubleMatrix a[] = new DoubleMatrix[W.length + 1];

		// z[] -> Weighted Sums
		DoubleMatrix z[] = new DoubleMatrix[W.length + 1];

		// Semi Forward pass to compute rhoHat
		for (int k = 0; k < a.length - 1; k++) {
			// k == 0 i.e. input layer units are the first activation units
			if (k == 0) {
				// Extracting i'th training example
				a[k] = X;
			} else {

				// Getting the weighted sum i.e. z = W.a + b
				z[k] = W[k - 1].mmul(a[k - 1]).addColumnVector(b[k - 1]);
				// Getting the activation by sigmoid function i.e. a = sigmoid(z)
				a[k] = Activations.sigmoid(z[k]);
			}
		}
		DoubleMatrix rhoHat = a[1].rowSums().div(m);
		z[a.length - 1] = W[a.length - 2].mmul(a[a.length - 2]).addColumnVector(b[a.length - 2]);
		a[a.length - 1] = z[a.length - 1];
		
		/*for (int k = 0; k < a.length; k++) {
			// k == 0 i.e. input layer units are the first activation units
			if (k == 0) {
				// Extracting i'th training example
				a[k] = X;
			} else {

				// Getting the weighted sum i.e. z = W.a + b
				z[k] = W[k - 1].mmul(a[k - 1]).addColumnVector(b[k - 1]);
				// Getting the activation by sigmoid function i.e. a = sigmoid(z)
				a[k] = Activations.sigmoid(z[k]);
			}
		}*/

		// Final activations are our hypothesis h
		// Dimensions of h : 10 x 5000
		DoubleMatrix h = a[a.length - 1];

		// Dimensions of Y : 5000 x 10
		J = CostFunctions.squaredError(X, h) / (2*m);
		// Regularizing cost
		J = J + ((lambda / 2) * MatrixUtil.squareSum(W));

		double sparsePenalty = KL(sparsityParam, rhoHat);
		J = J + (beta * sparsePenalty);

		DoubleMatrix sn[] = new DoubleMatrix[a.length];
		sn[sn.length - 1] = h.sub(X);

		DoubleMatrix sparsePenaltyDelta = KL_Delta(sparsityParam, rhoHat);

		for (int k = sn.length - 2; k > 0; k--) {
			sn[k] = (W[k].transpose().mmul(sn[k + 1])
					.addColumnVector(sparsePenaltyDelta.mul(beta))).mul(Activations
					.sigmoidGradient(z[k]));
		}
		for (int k = 0; k < W.length; k++) {
			W_grad[k] = sn[k + 1].mmul(a[k].transpose());
			b_grad[k] = sn[k + 1].rowSums();
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
		grad = MatrixUtil.rollParameters(W_grad, b_grad);

		CostGradient J_grad = new CostGradient(J, grad);
		// Creating an object to return cost along with gradient
		return J_grad;

	}
	
	// Reshapes nnParams into encoding and decoding parameters i.e. W and b
	public static Parameters reshape(DoubleMatrix nnParams, int visibleSize,
			int hiddenSize) {

		DoubleMatrix W[] = new DoubleMatrix[2];
		DoubleMatrix b[] = new DoubleMatrix[2];

		int start = 0;
		int endb = hiddenSize;
		int end = hiddenSize * (1 + visibleSize);
		b[0] = new DoubleMatrix(hiddenSize, 1, nnParams.getRange(start, endb)
				.toArray());
		W[0] = new DoubleMatrix(hiddenSize, visibleSize, nnParams.getRange(endb,
				end).toArray());

		start = end;
		endb = start + visibleSize;
		end = nnParams.rows;
		b[1] = new DoubleMatrix(visibleSize, 1, nnParams.getRange(start, endb)
				.toArray());
		W[1] = new DoubleMatrix(visibleSize, hiddenSize, nnParams.getRange(endb,
				end).toArray());

		return new Parameters(W, b);
	}
	
	// Gradient of KL Divergence
	public static DoubleMatrix KL_Delta(double rho, DoubleMatrix rhoHat) {
		
		// It calculates eq: KL_Delta = [-rho./rhoHat]  +  [(1-rho)./ (1-rhoHat)]
		DoubleMatrix c = rhoHat.rdiv(-rho);
		DoubleMatrix ones = DoubleMatrix.ones(rhoHat.rows, rhoHat.columns);
		DoubleMatrix d = ones.sub(rhoHat);
		d = d.rdiv(1 - rho);

		return c.add(d);

	}
	
	// Calculates KL Divergence
	public static double KL(double rho, DoubleMatrix rhoHat) {
		
		// It calculates the eq: KL = rho.* log (rho./rhoHat)  +  (1-rho).*log((1-rho)./(1-rhoHat))
		
		DoubleMatrix c = rhoHat.rdiv(rho);
		c = MatrixFunctions.log(c);
		c = c.mul(rho);

		DoubleMatrix ones = DoubleMatrix.ones(rhoHat.rows, rhoHat.columns);
		DoubleMatrix d = ones.sub(rhoHat);
		d = d.rdiv(1 - rho);
		d = MatrixFunctions.log(d);
		d = d.mul(1 - rho);

		c = c.add(d);

		DoubleMatrix sum = c.columnSums();

		return sum.get(0, 0);

	}

	public DoubleMatrix[] getW() {
		return W;
	}

	public void setW(DoubleMatrix[] w) {
		W = w;
	}

	public DoubleMatrix[] getB() {
		return b;
	}

	public void setB(DoubleMatrix[] b) {
		this.b = b;
	}

	// Trains the network for 'iter' no of iterations using backpropagation
	// algorithm
	public void training(int iter) {
		FminuncBatch.minimize(iter, this);
	}

	public DoubleMatrix getX() {
		return X;
	}

	public void setX(DoubleMatrix x) {
		X = x;
	}

	public int getM() {
		return m;
	}

	public void setM(int m) {
		this.m = m;
	}

	public DoubleMatrix getNnParams() {
		return nnParams;
	}

	public void setNnParams(DoubleMatrix nnParams) {
		this.nnParams = nnParams;
	}

}
