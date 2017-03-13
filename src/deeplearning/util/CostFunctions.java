package deeplearning.util;

/*
 * CostFunctions provides multiple static functions to evaluate cost
 * Currently, there is only logarithmic function
 * 
 * Written By: Muhammad Zaheer 
 * Dated : 5 Oct 2014
 */

import org.jblas.*;



public class CostFunctions {

	public static double logarithmic(DoubleMatrix y, DoubleMatrix h) {

		/*
		 * This function computes the logarithmic difference/cost For the hypothesis
		 * vector h against output vector y
		 * 
		 * cost = -y*log(h) - (1-y)* log(1-h)
		 */

		// firstTerm = -y*log(h)
		DoubleMatrix firstTerm = y.neg().mul(MatrixFunctions.log(h));

		// secondTerm = (1-y)*log(1-h)
		// Awkward technique to calculate (1-y) because of limitation of JBlas API
		DoubleMatrix ones = DoubleMatrix.ones(y.rows, y.columns);
		DoubleMatrix onesT = ones.transpose();

		DoubleMatrix secondTerm = ones.sub(y).mul(
				MatrixFunctions.log(onesT.sub(h)));
		
		/*
		 *  cost = firstTerm - secondTerm 
		 *  i.e. cost = -y*log(h) - (1-y) * log(1-h)
		 */
		
		double cost = firstTerm.rowSums().columnSums().get(0,0) - secondTerm.rowSums().columnSums().get(0,0);
		return cost;

	}
	
	public static double squaredError (DoubleMatrix y, DoubleMatrix h) {
		
		DoubleMatrix diff = y.sub(h);
		diff = diff.mul(diff);
		
		DoubleMatrix colSum = diff.columnSums();
		DoubleMatrix rowSum = colSum.rowSums();
		return rowSum.get(0,0);
		
	}

}
