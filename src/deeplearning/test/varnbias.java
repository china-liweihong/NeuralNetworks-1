package deeplearning.test;

import org.jblas.*;
import org.jblas.ranges.*;
import static org.jblas.DoubleMatrix.*;
import static org.jblas.MatrixFunctions.*;
import com.jmatio.types.*;
import com.jmatio.io.*;
import java.util.*;
import java.io.*;
import java.awt.Color;
import java.awt.BasicStroke;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;

public class varnbias {

	public DoubleMatrix X;
	public DoubleMatrix y;
	public DoubleMatrix Xval;
	public DoubleMatrix yval;
	public DoubleMatrix Xtest;
	public DoubleMatrix ytest;
	public DoubleMatrix costreg;
	public DoubleMatrix theta = DoubleMatrix.ones(2, 1);
	// thetainput minimize is used in minimization function
	public DoubleMatrix thetaminimize = DoubleMatrix.zeros(2, 1);
	public int m;
	public int n = theta.rows;
	public DoubleMatrix grad;
	public static DoubleMatrix errortrain;// =new DoubleMatrix(m,1);
	public static DoubleMatrix errorval;// =new DoubleMatrix(m,1);
	public DoubleMatrix trainingexamples;

	public static class plotgraph extends ApplicationFrame {
		public plotgraph(String applicationTitle, String chartTitle) {

			super(applicationTitle);
			JFreeChart xylineChart = ChartFactory.createXYLineChart(chartTitle,
					"No. of TrainingExamples", "Cost", createDataset(),
					PlotOrientation.VERTICAL, true, true, false);

			ChartPanel chartPanel = new ChartPanel(xylineChart);
			chartPanel.setPreferredSize(new java.awt.Dimension(800, 1000));
			final XYPlot plot = xylineChart.getXYPlot();
			XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
			renderer.setSeriesPaint(0, Color.RED);
			renderer.setSeriesPaint(1, Color.GREEN);

			renderer.setSeriesStroke(0, new BasicStroke(4.0f));
			renderer.setSeriesStroke(1, new BasicStroke(3.0f));

			plot.setRenderer(renderer);
			setContentPane(chartPanel);

		}

		private XYDataset createDataset() {
			final XYSeries train = new XYSeries("Train");

			for (int x = 0; x < errortrain.rows; x++) {
				train.add((double) x + 1, errortrain.get(x, 0));
			}
			// train.add( 1.0 , 1.0 );
			// train.add( 2.0 , 4.0 );
			// train.add( 3.0 , 3.0 );
			final XYSeries crossval = new XYSeries("CrossValidation");

			for (int x = 0; x < errorval.rows; x++) {
				crossval.add((double) x + 1, errorval.get(x, 0));
			}

			// crossval.add( 1.0 , 4.0 );

			// crossval.add( 2.0 , 5.0 );

			// crossval.add( 3.0 , 6.0 );

			final XYSeriesCollection dataset = new XYSeriesCollection();
			dataset.addSeries(train);
			dataset.addSeries(crossval);

			return dataset;
		}
	}

	public void loadData(String dataFile) {
		MatFileReader mfr = null;

		try {
			mfr = new MatFileReader(dataFile);
		} catch (IOException e) {
			e.printStackTrace();
		}

		MLDouble mlArrX = (MLDouble) (mfr.getMLArray("X"));
		MLDouble mlArry = (MLDouble) (mfr.getMLArray("y"));
		MLDouble mlArrXval = (MLDouble) (mfr.getMLArray("Xval"));
		MLDouble mlArryval = (MLDouble) (mfr.getMLArray("yval"));
		MLDouble mlArrXtest = (MLDouble) (mfr.getMLArray("Xtest"));
		MLDouble mlArrytest = (MLDouble) (mfr.getMLArray("ytest"));

		int dimX[] = mlArrX.getDimensions();
		int dimY[] = mlArry.getDimensions();
		int dimXval[] = mlArrXval.getDimensions();
		int dimYval[] = mlArryval.getDimensions();
		int dimXtest[] = mlArrXtest.getDimensions();
		int dimYtest[] = mlArrytest.getDimensions();

		double[][] matArrX = new double[dimX[0]][dimX[1]];
		double[][] matArry = new double[dimY[0]][dimY[1]];
		double[][] matArrXval = new double[dimXval[0]][dimXval[1]];
		double[][] matArryval = new double[dimYval[0]][dimYval[1]];
		double[][] matArrXtest = new double[dimXtest[0]][dimXtest[1]];
		double[][] matArrytest = new double[dimYtest[0]][dimYtest[1]];

		matArrX = mlArrX.getArray();
		matArry = mlArry.getArray();
		matArrXval = mlArrXval.getArray();
		matArryval = mlArryval.getArray();
		matArrXtest = mlArrXtest.getArray();
		matArrytest = mlArrytest.getArray();

		this.X = new DoubleMatrix(matArrX);
		this.y = new DoubleMatrix(matArry);
		this.Xval = new DoubleMatrix(matArrXval);
		this.yval = new DoubleMatrix(matArryval);
		this.Xtest = new DoubleMatrix(matArrXtest);
		this.ytest = new DoubleMatrix(matArrytest);

		this.m = X.rows;

		// System.out.println(X.rows);

	}

	public void computecostreg(DoubleMatrix x, DoubleMatrix y,
			DoubleMatrix thetainput, int lambda) {
		DoubleMatrix firstterm = new DoubleMatrix(1, 1);
		DoubleMatrix curr;
		DoubleMatrix bias;
		DoubleMatrix curry;
		DoubleMatrix subft;
		int size = x.rows;

		for (int x1 = 0; x1 < size; x1++) {

			curr = x.getRow(x1).transpose();

			bias = DoubleMatrix.ones(1, 1);

			curr = curr.concatVertically(bias, curr);
			curr = (thetainput.transpose()).mmul(curr);

			curry = y.getRow(x1).transpose();
			subft = curr.sub(curry);
			firstterm = firstterm.add(subft.mmul(subft.transpose()));

		}
		firstterm = firstterm.div(2 * size);

		DoubleMatrix secondterm;
		DoubleMatrix currt;
		DoubleMatrix sqrd = new DoubleMatrix(1, 1);
		for (int y1 = 0; y1 < n - 1; y1++) {

			currt = thetainput.getRow(y1 + 1);
			sqrd = sqrd.add(currt.mmul(currt.transpose()));

		}

		secondterm = sqrd.mul((double) lambda / (double) (2 * size));
		costreg = firstterm.add(secondterm);

		DoubleMatrix Xwb;
		DoubleMatrix bias1;
		DoubleMatrix htheta;

		// gradients from here

		bias1 = DoubleMatrix.ones(size, 1);
		// System.out.println("Asd"+ "X rows " +x.rows+" biasrows "+ bias1.rows);

		Xwb = concatHorizontally(bias1, x);

		double a;
		for (int x2 = 0; x2 < size; x2++) {
			a = Xwb.get(x2, 0);
			// System.out.println(a);
		}

		htheta = Xwb.mmul(thetainput);
		grad = (Xwb.transpose().mmul(htheta.sub(y))).div((double) size);
		double t = (double) lambda / (double) size;
		// System.out.println(t);
		for (int y2 = 0; y2 < n - 1; y2++) {
			DoubleMatrix currgrad;
			DoubleMatrix currtheta;
			currtheta = thetainput.getRow(y2 + 1);
			currgrad = grad.getRow(y2 + 1);

			currgrad.add(currtheta.mmul(t));
			grad.putRow(y2 + 1, currgrad);
			// System.out.println(grad.get(y2,0));
			// System.out.println(grad.get(y2+1,0));

		}

	}

	public void minimize(DoubleMatrix X1, DoubleMatrix y1, int length) {

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
		int datasize = X1.rows;
		// varnbias

		DoubleMatrix input = thetaminimize;
		int M = 0;
		int i = 0; // zero the run length counter
		int initial_step = 1; // starting point
		int ls_failed = 0; // no previous line search has failed
		// CostGradient cg;

		theta = input;
		// cg = this.nnCostFunction();
		computecostreg(X1, y1, theta, 0);
		double f1 = costreg.get(0, 0);
		DoubleMatrix df1 = grad;

		i = i + (length < 0 ? 1 : 0);

		DoubleMatrix s = df1.neg(); // searchdirection (opposite of gradient)
		double d1 = s.neg().dot(s); // the slope
		double z1 = initial_step / (1.0 - d1); // initial step
		while (i < Math.abs(length)) { // while loop for total iterations

			i = i + (length > 0 ? 1 : 0); // count iterations

			// copy of current values

			DoubleMatrix X0 = input.dup();
			double f0 = f1;
			DoubleMatrix df0 = df1.dup();

			// begin line search

			input = input.add(s.mul(z1)); // incrementing thetainput by
																		// thetainput+(step*searchdirection)

			// now calculating gradient and cost with respect to new thetainput
			theta = input;
			// cg = this.nnCostFunction();
			computecostreg(X1, y1, theta, 0);
			double f2 = costreg.get(0, 0);
			DoubleMatrix df2 = grad;

			i = i + (length < 0 ? 1 : 0); // count epochs
			double d2 = df2.dot(s);

			// initialize point 3 equal to point 1

			double f3 = f1;
			double d3 = d1;
			double z3 = -z1;
			if (length > 0) {
				M = MAX;
			} else {
				M = Math.min(MAX, -length - i);
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

					// Double.isNANA and Double.isinifinite are not known functions to me!

					if (Double.isNaN(z2) || Double.isInfinite(z2)) {
						// if we had a numerical problem then bisect
						z2 = z3 / 2.0d;
					}
					// don't accept too close to limits
					z2 = Math.max(Math.min(z2, INT * z3), (1 - INT) * z3);
					// update the step
					z1 = z1 + z2;
					input = input.add(s.mul(z2));
					theta = input;
					// cg = this.nnCostFunction();
					computecostreg(X1, y1, theta, 0);
					f2 = costreg.get(0, 0);
					df2 = grad;

					M = M - 1;
					i = i + (length < 0 ? 1 : 0); // count epochs
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
				theta = input;
				// cg = this.nnCostFunction();
				computecostreg(X1, y1, theta, 0);
				f2 = costreg.get(0, 0);
				df2 = grad;

				M = M - 1;
				i = i + (length < 0 ? 1 : 0); // count epochs?!
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
				if (ls_failed == 1 || i > Math.abs(length)) {
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

	}

	public void crossvalerror() {
		DoubleMatrix trainingdataX = null;
		DoubleMatrix trainingdatay = null;
		DoubleMatrix adddataX;
		DoubleMatrix adddatay;
		DoubleMatrix temp;
		System.out.println("TrainingExamples    CostTrain      CostVal");
		adddatay = y.getRow(0);
		adddataX = X.getRow(0);
		int trainingsize;
		int trainingsizeval = Xval.rows;

		for (int i = 0; i < m; i++) {
			System.out.print(i + 1);
			if (i == 0) {
				trainingdataX = adddataX;
				trainingdatay = adddatay;
				trainingsize = trainingdataX.rows;
				minimize(trainingdataX, trainingdatay, 200);
				computecostreg(trainingdataX, trainingdatay, theta, 0);
				temp = costreg.getRow(0);
				errortrain = temp;
				System.out.print("               " + errortrain.getRow(i));
				computecostreg(Xval, yval, theta, 0);
				temp = costreg.getRow(0);
				errorval = temp;
				System.out.print("        " + errorval.getRow(i));
				System.out.println(" ");
			} else {

				adddatay = y.getRow(i);
				adddataX = X.getRow(i);
				trainingdataX = concatVertically(trainingdataX, adddataX);
				trainingdatay = concatVertically(trainingdatay, adddatay);
				trainingsize = trainingdataX.rows;
				minimize(trainingdataX, trainingdatay, 200);
				computecostreg(trainingdataX, trainingdatay, theta, 0);
				temp = costreg.getRow(0);
				// errortrain.putRow(i, temp);
				errortrain = concatVertically(errortrain, temp);
				System.out.print("               " + errortrain.getRow(i));
				computecostreg(Xval, yval, theta, 0);
				temp = costreg.getRow(0);
				// errorval.putRow(i, temp);
				errorval = concatVertically(errorval, temp);
				System.out.print("        " + errorval.getRow(i));
				System.out.println(" ");
			}
			// theta=minimize()
		}
	}

	public static void main(String[] args) {

		varnbias vnb = new varnbias();

		// Data file to be read
		String dataFile = "data/ex5data1.mat";

		vnb.loadData(dataFile);

		vnb.computecostreg(vnb.X, vnb.y, vnb.theta, 1);

		System.out.println("Reg cost is :" + vnb.costreg.get(0, 0));

		vnb.minimize(vnb.X, vnb.y, 200);

		System.out.println("Reg cost of trained network is :"
				+ vnb.costreg.get(0, 0));

		vnb.crossvalerror();

		plotgraph chart = new plotgraph("Bias Variance Check",
				"Training Error vs CrossValidation Error");
		chart.pack();
		RefineryUtilities.centerFrameOnScreen(chart);
		chart.setVisible(true);

	}

}