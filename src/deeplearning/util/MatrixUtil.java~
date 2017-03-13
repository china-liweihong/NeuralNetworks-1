package deeplearning.util;

import org.jblas.DoubleMatrix;

/*
 * MatrixUtil is a useful class which provides a set of utility functions
 * built on top of JBlas API.
 * 
 * Example: 
 * 1. Rolling parameters into a vector
 * 2. Randomly Initializing Weight matrices by adding a bias
 * 3. Printing dimensions of a matrix etc
 * 
 * Written By: Muhammad Zaheer
 * Dated: 03-Oct-2014
 */

public class MatrixUtil {
	// Creates a randomly initialized parameter/weight matrix
	// Dimension L_out * (1+L_in)
	// Adding 1 to compensate for the bias term
	public static DoubleMatrix randInitializeWeightsNew(int L_in, int L_out) {

		double epsilon_init = 0.12;
		DoubleMatrix W = DoubleMatrix.rand(L_out, 1 + L_in);
		W = W.mul(2 * epsilon_init);
		W = W.sub(epsilon_init);

		return W;
	}
	
	// New rand Initializer, which does not add bias unit
	public static DoubleMatrix randInitializeWeights(int L_in, int L_out) {

		double epsilon_init = 0.12;
		DoubleMatrix W = DoubleMatrix.rand(L_out, L_in);
		W = W.mul(2 * epsilon_init);
		W = W.sub(epsilon_init);

		return W;
	}
	
	public static DoubleMatrix randInitializeConv(int L_in, int L_out)	{
		
		double r = Math.sqrt(6);
		r = r/ Math.sqrt(L_out+L_in+1);
		
		DoubleMatrix Wd = DoubleMatrix.rand(L_in, L_out);
		Wd = Wd.mul(2).mul(r).sub(r);
		
		return Wd;
	}
	// Rolls matrix a and b into a single parameter
	// Column wise
	public static DoubleMatrix rollParameters(DoubleMatrix a, DoubleMatrix b) {

		DoubleMatrix c = new DoubleMatrix(a.rows * a.columns, 1, a.data);
		c = DoubleMatrix.concatVertically(c, new DoubleMatrix(b.rows * b.columns,
				1, b.data));
		return c;

	}
	
	public static DoubleMatrix rollParameters (DoubleMatrix a, DoubleMatrix b,DoubleMatrix c) {
		DoubleMatrix d = new DoubleMatrix (a.rows*a.columns,1,a.data);
		
		d = DoubleMatrix.concatVertically(d , new DoubleMatrix(b.rows*b.columns,1,b.data));
		
		d = DoubleMatrix.concatVertically(d , new DoubleMatrix(c.rows*c.columns,1,c.data));
		
		return d;
		
	}
	
	public static DoubleMatrix rollParameters (DoubleMatrix a[]){
		DoubleMatrix b = new DoubleMatrix(a[0].rows*a[0].columns,1,a[0].data);
		
		for (int i = 1; i < a.length;i++) {
			b = DoubleMatrix.concatVertically(b, new DoubleMatrix(a[i].rows*a[i].columns,1,a[i].data));
		}
		
		return b;
	}
	
	public static DoubleMatrix rollParameters( DoubleMatrix W[], DoubleMatrix b[]) {
		DoubleMatrix c = new DoubleMatrix(b[0].rows*b[0].columns,1,b[0].data);
		c = DoubleMatrix.concatVertically(c, new DoubleMatrix(W[0].rows * W[0].columns,1,W[0].data));
		
		for (int i = 1; i < W.length; i++) {
			c = DoubleMatrix.concatVertically(c, new DoubleMatrix(b[i].rows * b[i].columns,1,b[i].data));
			c = DoubleMatrix.concatVertically(c, new DoubleMatrix(W[i].rows * W[i].columns,1,W[i].data));
			
			
		}
		
		return c;
		
	}
	
	
	// Prints a given matrix
	public static void printMatrix(DoubleMatrix mat) {
		for (int i = 0; i < mat.rows; i++) {
			for (int j = 0; j < mat.columns; j++)
				System.out.print(mat.get(i, j) + " ");

			System.out.println();

		}
	}

	// printDimensions of the given matrix
	// Useful for debugging
	public static void printDimensions(DoubleMatrix mat) {

		System.out.println("Rows: " + mat.rows);
		System.out.println("Cols: " + mat.columns);

	}
	
	public static double squareSum (DoubleMatrix mat[]) {
		
		double sum = 0;
		for (int i = 0; i< mat.length;i++) {
			// Making a duplicate because element wise multiply is in place
			DoubleMatrix tempW = mat[i].dup();
			DoubleMatrix sqW = tempW.mul(tempW);
			DoubleMatrix colSum = sqW.columnSums();
			DoubleMatrix rowSum = colSum.rowSums();
			
			sum = sum + rowSum.get(0,0);
			
		}
		return sum;
	}

	public static double[][] downsample(double[][] convOut, int convDim, int PoolDim) {
		// TODO Auto-generated method stub
		
		double[][] image = convOut;
		double[][] downsampled = new double[convDim/PoolDim][convDim/PoolDim]; 
		int size = convDim/PoolDim;
		
		for( int i = 0 ; i < size ; i++)	{
			for ( int j = 0; j < size; j++ )	{
				
				downsampled[i][j] = image[i*PoolDim][j*PoolDim];
			}
		}
		return downsampled;
	}

}
