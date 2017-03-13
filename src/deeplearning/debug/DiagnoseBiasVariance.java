package deeplearning.debug;

import java.awt.BasicStroke;
import java.awt.Color;

import org.jblas.*;
import org.jblas.ranges.IntervalRange;
import org.jblas.util.Permutations;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import deeplearning.neuralnetwork.*;
import deeplearning.test.varnbias.plotgraph;
import deeplearning.util.CostGradient;
import deeplearning.util.MatrixUtil;

/*
 * Written By: Salman Saeed | Muhammad Zaheer
 * 
 * Dated : 30 Oct, 2014
 */

public class DiagnoseBiasVariance {

	private DoubleMatrix X, y;
	private DoubleMatrix XRaw, yRaw;

	private DoubleMatrix xTrain, yTrain, xVal, yVal, xTest, yTest;
	public NeuralNetwork nn;
	
	private int trainSize, valSize, testSize;
	private static DoubleMatrix errorTrain;
	private static DoubleMatrix errorVal;
	private static int step; 
	/*double [] errorTrain;
	double [] errorVal;
*/
	public void init(String dataFile) {

		this.nn.init(dataFile);

		this.XRaw = nn.getX();
		this.yRaw = nn.getY();

		this.shuffle();

	}

	public void shuffle() {
		int permutation[] = Permutations.randomPermutation(XRaw.rows);

		this.X = new DoubleMatrix(XRaw.rows, XRaw.columns);
		this.y = new DoubleMatrix(yRaw.rows, yRaw.columns);

		for (int i = 0; i < permutation.length; i++) {

			X.putRow(i, XRaw.getRow(permutation[i]));
			y.putRow(i, yRaw.getRow(permutation[i]));
		}

	}

	public void splitDataSet(int trainSize,
			int valSize, int testSize) {
		IntervalRange rowRangeX, colRangeX, rowRangeY, colRangeY;
		
		this.trainSize = trainSize;
		this.valSize = valSize;
		this.testSize = testSize;
		rowRangeX = new IntervalRange(0, trainSize);
		colRangeX = new IntervalRange(0, X.columns);
		rowRangeY = new IntervalRange(0, trainSize);
		colRangeY = new IntervalRange(0, y.columns);

		this.xTrain = X.get(rowRangeX, colRangeX);
		this.yTrain = y.get(rowRangeY, colRangeY);
		errorTrain = DoubleMatrix.zeros(trainSize,1);
		errorVal = DoubleMatrix.zeros (trainSize,1);
		rowRangeX = new IntervalRange(trainSize, trainSize + valSize);
		rowRangeY = new IntervalRange(trainSize, trainSize + valSize);

		this.xVal = X.get(rowRangeX, colRangeX);
		this.yVal = y.get(rowRangeY, colRangeY);

		rowRangeX = new IntervalRange(trainSize + valSize, trainSize + valSize
				+ testSize);
		rowRangeY = new IntervalRange(trainSize + valSize, trainSize + valSize
				+ testSize);

		this.xTest = X.get(rowRangeX, colRangeX);
		this.yTest = y.get(rowRangeY, colRangeY);

	}
	
	public void printDimensions() {
	
		
		MatrixUtil.printDimensions(this.xTrain);
		MatrixUtil.printDimensions(this.yTrain);
		MatrixUtil.printDimensions(this.xVal);
		MatrixUtil.printDimensions(this.yVal);
		MatrixUtil.printDimensions(this.xTest);
		MatrixUtil.printDimensions(this.yTest);
		
	}

	public void errorTrainVal(int step, int iter) {
		
		this.step = step;
		CostGradient J_grad;
		
		int start = 0;
		int end = step;
		
		
		IntervalRange rowRangeX = new IntervalRange (start,end);
		IntervalRange colRangeX = new IntervalRange (0,X.columns);
		
		IntervalRange rowRangeY = new IntervalRange (start,end);
		IntervalRange colRangeY = new IntervalRange (0, y.columns);
		for(int i = 0; i < (this.trainSize/step) ; i++) {
			System.out.println("i: " + i + " Start :" + start + " End: " + end);
			
			DoubleMatrix X = xTrain.get(rowRangeX, colRangeX);
			DoubleMatrix y = yTrain.get(rowRangeY, colRangeY);
			
			nn.setX(X);
			nn.setY(y);
			
			nn.training(iter);
			
			J_grad = nn.nnCostFunction();
			
			errorTrain.put(i,J_grad.getCost());
			
			J_grad = nn.nnCostFunction(xVal, yVal);
			
			errorVal.put(i,J_grad.getCost());
			
			end = end + step;
			rowRangeX = new IntervalRange (start, end);
			rowRangeY = new IntervalRange (start,end);
			nn.init();
		}
		
	}
	
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

			for (int x = 0; x < (errorTrain.rows/step); x++) {
				train.add((double) x + 1, errorTrain.get(x, 0));
			}
			final XYSeries crossval = new XYSeries("CrossValidation");

			for (int x = 0; x < (errorVal.rows/step); x++) {
				crossval.add((double) x + 1, errorVal.get(x, 0));
			}

			final XYSeriesCollection dataset = new XYSeriesCollection();
			dataset.addSeries(train);
			dataset.addSeries(crossval);

			return dataset;
		}
	}
	public static void main(String[] args) {

		String dataFile = "data/ex4data1.mat";

		DiagnoseBiasVariance dbv = new DiagnoseBiasVariance();

		int hiddenLayerSize[] = new int[] { 25 };

		dbv.nn = new NeuralNetwork(400, hiddenLayerSize, 10, 0);
		dbv.init(dataFile);
		
		dbv.splitDataSet(3000, 1000,1000);
		dbv.nn.setLambda(0.1);
		dbv.errorTrainVal(100, 50);
		
		plotgraph chart = new plotgraph("Bias Variance Check",
				"Training Error vs CrossValidation Error");
		chart.pack();
		RefineryUtilities.centerFrameOnScreen(chart);
		chart.setVisible(true);
		
	}

}
