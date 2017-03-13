package deeplearning.minimizers;

import org.jblas.DoubleMatrix;

import deeplearning.util.CostGradient;

/*
 * Minimization function which uses the costfunction of a given network
 * to minimze the parameters
 * 
 * Written By: Haider Adeel Agha
 * 
 */


public class FminuncBatch {
	// minimization function using matlab fmincg to minimize nonlinear
	// differentiable
	// fucntion using conjugate gradient descent and line search

	public static DoubleMatrix minimize(int iterations,Network costFunction) {

		double EXT = 3.0;
		// a bunch of constants for line searches
		double RHO = 0.01;
		// RHO and SIG are the constants in the Wolfe-Powell conditions
		double SIG = 0.5;
		// don't reevaluate within 0.1 of the limit of the current bracket
		double INT = 0.1;
		// max 20 function evaluations per line search
		int MAX = 20;
		// maximum allowed slope ratio
		int RATIO = 100;

		DoubleMatrix input = costFunction.getNnParams();
		int M = 0;
		int i = 0; // zero the run length counter
		int initial_step = 1; // starting point
		int ls_failed = 0; // no previous line search has failed
		CostGradient cg;
		
		costFunction.setNnParams(input);
		cg = costFunction.nnCostFunction();
		double f1 = cg.getCost();
		DoubleMatrix df1 = cg.getGradient();

		i = i + (iterations < 0 ? 1 : 0);

		DoubleMatrix s = df1.neg(); // searchdirection (opposite of gradient)
		double d1 = s.neg().dot(s); // the slope
		double z1 = initial_step / (1.0 - d1); // initial step
		while (i < Math.abs(iterations)) { // while loop for total iterations

			i = i + (iterations > 0 ? 1 : 0); // count iterations

			// copy of current values

			DoubleMatrix X0 = input.dup();
			double f0 = f1;
			DoubleMatrix df0 = df1.dup();

			// begin line search

			input = input.add(s.mul(z1)); // incrementing theta by
			// theta+(step*searchdirection)

			// now calculating gradient and cost with respect to new theta
			costFunction.setNnParams(input);
			cg = costFunction.nnCostFunction();
			System.out.print("i: ");
			System.out.format("%3d",i );
			System.out.println(" | Cost: " + cg.getCost());
			double f2 = cg.getCost();
			DoubleMatrix df2 = cg.getGradient();

			i = i + (iterations < 0 ? 1 : 0); // count epochs
			double d2 = df2.dot(s);

			// initialize point 3 equal to point 1

			double f3 = f1;
			double d3 = d1;
			double z3 = -z1;
			if (iterations > 0) {
				M = MAX;
			} else {
				M = Math.min(MAX, -iterations - i);
			}

			// initialize success and limit

			int success = 0;
			double limit = -1;

			while (true) {
				while (((f2 > f1 + z1 * RHO * d1) | (d2 > -SIG * d1)) && (M > 0)) {

					// tighten the bracket
					limit = z1;
					double z2 = 0.0d;
					double A = 0.0d;
					double B = 0.0d;

					if (f2 > f1) {
						// quadratic fit
						z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3);
					} else {
						// cubic fit
						A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
						B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
						// numerical error possible
						z2 = (Math.sqrt(B * B - A * d2 * z3 * z3) - B) / A;
					}

					// Double.isNANA and Double.isinifinite are not known
					// functions to me!

					if (Double.isNaN(z2) || Double.isInfinite(z2)) {
						// if we had a numerical problem then bisect
						z2 = z3 / 2.0d;
					}
					// don't accept too close to limits
					z2 = Math.max(Math.min(z2, INT * z3), (1 - INT) * z3);
					// update the step
					z1 = z1 + z2;
					input = input.add(s.mul(z2));
					costFunction.setNnParams(input);
					cg = costFunction.nnCostFunction();
					f2 = cg.getCost();
					df2 = cg.getGradient();

					M = M - 1;
					i = i + (iterations < 0 ? 1 : 0); // count epochs
					d2 = df2.dot(s);
					// z3 is now relative to the location of z2
					z3 = z3 - z2;
				}

				if (f2 > f1 + z1 * RHO * d1 || d2 > -SIG * d1) {
					break; // failure
				} else if (d2 > SIG * d1) {
					success = 1;
					break; // success
				} else if (M == 0) {
					break; // failure
				}
				// make cubic extrapolation
				double A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
				double B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
				double z2 = -d2 * z3 * z3 / (B + Math.sqrt(B * B - A * d2 * z3 * z3));

				// again check here
				if (Double.isNaN(z2) || Double.isInfinite(z2) || z2 < 0) {
					// if we have no upper limit
					if (limit < -0.5) {
						// the extrapolate the maximum amount
						z2 = z1 * (EXT - 1);
					} else {
						// otherwise bisect
						z2 = (limit - z1) / 2;
					}
				}

				else if ((limit > -0.5) && (z2 + z1 > limit)) {
					// extraplation beyond max?
					z2 = (limit - z1) / 2; // bisect
				} else if ((limit < -0.5) && (z2 + z1 > z1 * EXT)) {
					// extrapolationbeyond limit
					z2 = z1 * (EXT - 1.0); // set to extrapolation limit
				} else if (z2 < -z3 * INT) {
					z2 = -z3 * INT;
				} else if ((limit > -0.5) && (z2 < (limit - z1) * (1.0 - INT))) {
					// too close to the limit
					z2 = (limit - z1) * (1.0 - INT);
				}

				// set point 3 equal to point 2
				f3 = f2;
				d3 = d2;
				z3 = -z2;
				z1 = z1 + z2;
				// update current values
				input = input.add(s.mul(z2));
				costFunction.setNnParams(input);
				cg = costFunction.nnCostFunction();
				f2 = cg.getCost();
				df2 = cg.getGradient();

				M = M - 1;
				i = i + (iterations < 0 ? 1 : 0); // count epochs?!
				d2 = df2.dot(s);
			}
			// end of line search
			DoubleMatrix tmp = null;
			if (success == 1) { // if line search succeeded
				f1 = f2;

				// Polack-Ribiere direction: s =
				// (df2'*df2-df1'*df2)/(df1'*df1)*s - df2;
				final double numerator = (df2.dot(df2) - df1.dot(df2)) / df1.dot(df1);
				s = s.mul(numerator).sub(df2);
				tmp = df1;
				df1 = df2;
				df2 = tmp; // swap derivatives
				d2 = df1.dot(s);
				if (d2 > 0) { // new slope must be negative
					s = df1.neg(); // otherwise use steepest direction
					d2 = s.neg().dot(s);
				}
				// realmin in octave = 2.2251e-308
				// slope ratio but max RATIO
				z1 = z1 * Math.min(RATIO, d1 / (d2 - 2.2251e-308));
				d1 = d2;
				ls_failed = 0; // this line search did not fail

			} else {

				input = X0;
				f1 = f0;
				df1 = df0; // restore point from before failed line search
				// line search failed twice in a row?
				if (ls_failed == 1 || i > Math.abs(iterations)) {
					break; // or we ran out of time, so we give up
				}
				tmp = df1;
				df1 = df2;
				df2 = tmp; // swap derivatives
				s = df1.neg(); // try steepest
				d1 = s.neg().dot(s);
				z1 = 1.0d / (1.0d - d1);
				ls_failed = 1; // this line search failed
			}
		}
		return input;
	}

}
