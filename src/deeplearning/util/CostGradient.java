package deeplearning.util;

import org.jblas.DoubleMatrix;

/*
 * CostGradient provides a grad vector and corresponding cost
 * The object of this class is used by the minimization function
 * 
 * Written By: Muhammad Zaheer
 * Dated: 03-Oct-2014
 */

public class CostGradient {
	private Double cost;
	private DoubleMatrix grad;

	public CostGradient() {
	}

	public CostGradient(double cost, DoubleMatrix grad) {
		this.cost = cost;
this.grad = grad;
	}

	public double getCost() {
		return this.cost;
	}

	public DoubleMatrix getGradient() {
		return this.grad;
	}

	public void setCost(double cost) {
		this.cost = cost;
		return;
	}

	public void setGradient(DoubleMatrix grad) {
		this.grad = grad;
		return;
	}

	public void setCostGradient(double cost, DoubleMatrix grad) {
		this.cost = cost;
		this.grad = grad;
		return;
	}
}
