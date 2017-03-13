package deeplearning.convolutional;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.concurrent.CyclicBarrier;

import org.jblas.DoubleMatrix;
import deeplearning.neuralnetwork.*;
import deeplearning.util.CostGradient;
import deeplearning.util.MatrixUtil;
//import deeplearning.convolutional.Convolutional;
import deeplearning.convolutional.Cifar;
/* Checks the implementation of Backpropagation Algorithm for Neural Network
*/

public class CheckConvGradients {

	
	int [] filterDim = new int[]{5,5};
	int [] numOfFilters = new int[]{2,2};
	int [] inputDim = new int[]{28,12};
	int [] channels = new int[]{1,2};
	int [] padding = new int[]{0,4};
	int [] poolDim = new int[]{2,2};
	int layers = 2;
	
	
	public Cifar convo;
	public Convolutional cnn;
	
	public CheckConvGradients() {

	
		//cnn = new Convolutional(28,10,filterDim,numOfFilters,poolDim,inputDim, channels, layers);
		cnn = new Convolutional();
		cnn.numImages = 10;
		cnn.colImages = new DoubleMatrix[cnn.numImages][channels[0]];
		//Convolutional.Labels = DoubleMatrix.zeros(cnn.numImages, 1);
		try {
			cnn.loadCSV("data/mnist_train.csv", 10);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public CheckConvGradients(int imageDim, int numClasses, int filterDim, int numOfFilters,
			int poolDim) {

		//cnn = new Convolutional(32,numClasses,filterDim,2,5);
		//convo = new ConvNetCost(28,10,9,2,5);
		
	}

	public void check() throws IOException {
		
		
		
//		cnn.initParameters();
//
//		//convo = new Cifar(cnn.colImages, cnn.Labels, cnn.nnParams, 28, 10, filterDim, numOfFilters,
//		//		poolDim, channels, layers, padding, "cifar");	
//		
//		DoubleMatrix numGrad = computeNumericalGradients(cnn.nnParams);
//		
//		CostGradient backprop = convo.Cost(Convolutional.colImages, Convolutional.Labels, cnn.nnParams);
//		DoubleMatrix back = backprop.getGradient();
//		
//		double sum= 0.0;
//		
//		int filters = 0;
//		int filter = 0;
//		filter = filterDim[0]*filterDim[0]*channels[0]*numOfFilters[0];
//		for( int i = 0 ; i < filterDim.length ; i++){
//			filters += filterDim[i]*filterDim[i]*channels[i]*numOfFilters[i];
//		}
//		
//		for (int i =0; i<numGrad.length;i++) {
//			
//			System.out.println (back.get(i) + " " + numGrad.get(i) + " " + ((Math.abs(back.get(i) - numGrad.get(i)))/numGrad.get(i))*100);
//			if( i == filters || i == filter){
//				System.out.println();System.out.println();System.out.println();
//			}
//			sum = sum + Math.abs(back.get(i) - numGrad.get(i));
//			}
//		
//		System.out.println ("Norm diff: " +sum);
	}
		
	
	public DoubleMatrix computeNumericalGradients(DoubleMatrix theta) {
		
//		DoubleMatrix numGrad = DoubleMatrix.zeros (theta.rows,theta.columns);
//		DoubleMatrix perturb = DoubleMatrix.zeros(theta.rows,theta.columns);
//		
//		double e = 0.00001;
//		
//		for (int p = 0; p < (theta.rows*theta.columns); p++) {
//			perturb.put(p, e);
//			
//			DoubleMatrix tempo = theta.add(perturb);
//			CostGradient J = convo.Cost(cnn.colImages, cnn.Labels, tempo);
//			double J1 = J.getCost();
//			
//			tempo = theta.sub(perturb);
//			J = convo.Cost(Convolutional.colImages, Convolutional.Labels, tempo);
//			double J2 = J.getCost();
//			
//			double temp = (J1 - J2)/ (2*e); 
//			numGrad.put(p,temp);
//			perturb.put(p, 0);
//		}
//		
//		
//		
//		return numGrad;
		return new DoubleMatrix();
	}	
	
	public static void main(String args[]) throws IOException {
		
		CheckConvGradients c = new CheckConvGradients();
//		
//		cnn = new Convolutional();
//		
		c.check();

	}

}

