package deeplearning.util;

import org.jblas.*;

/*
 * Activations class provides a set of static functions for activations
 * as well as there respective gradients
 * Activation could be one of sigmoid, tan, etc.
 * 
 * Right now, we have implemented just sigmoid function and its gradient
 * 
 * Written By: Muhammad Zaheer
 * Dated: 03-Oct-2014
 */

public class Activations {
	

	
	// Computes sigmoid of the given scalar, vector or matrix
	public static DoubleMatrix sigmoid(DoubleMatrix z) {

		z = z.mul(-1);
		DoubleMatrix e = MatrixFunctions.exp(z);
		e = e.add(1);
		DoubleMatrix g = DoubleMatrix.ones(e.rows, e.columns);
		g = g.div(e);
		return g;
	
	}

	// Calculates the sigmoid gradient of the given scalar, vector or matrix
	public static DoubleMatrix sigmoidGradient(DoubleMatrix z) {

		DoubleMatrix ones = DoubleMatrix.ones(z.rows, z.columns);

		DoubleMatrix g = sigmoid(z).mul(ones.sub(sigmoid(z)));

		return g;

	}
}
