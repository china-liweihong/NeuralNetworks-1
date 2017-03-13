package deeplearning.minimizers;

import deeplearning.util.CostGradient;

import org.jblas.*;

/*
 * An interface to assist a minimizer getting access to the cost function
 * implemented by a particular network as well as updating/getting the parameters
 * 
 * Written By: Muhammad Zaheer
 * 
 */

public interface Network {
	public CostGradient nnCostFunction();

	public DoubleMatrix getNnParams();

	public void setNnParams(DoubleMatrix nn_params);
}
