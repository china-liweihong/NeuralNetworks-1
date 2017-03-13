package deeplearning.runtime;

import org.jblas.DoubleMatrix;
import deeplearning.display.Display;

import deeplearning.sparseautoencoder.*;
import deeplearning.util.CostGradient;
import deeplearning.util.Parameters;
public class RunSparseAutoencoder {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		double lambda = 0.0001;
		double beta = 3;
		double sparsityParam = 0.01;
		SparseAutoencoder sa = new SparseAutoencoder(64,25,lambda,sparsityParam,beta);

		sa.loadData("data/IMAGES.mat");
		sa.generateTrainingSet(1000, 8);
		
		
		sa.initParameters();
		CostGradient cg = sa.nnCostFunction();
		
		System.out.println("Init Cost: " + cg.getCost());
		//	Training the network for 100 iterations
		sa.training(400);
		
		
		
		Parameters pm  = SparseAutoencoder.reshape(sa.getNnParams(), 64, 25);
		
		DoubleMatrix W[] = pm.getW();
		
		//MatrixUtil.printDimensions(W[0]);
		//MatrixUtil.printMatrix(W[0]);
		
		
		Display disp = new Display();
		disp.DisplayNetwork(W[0]);
		
		//cg = sa.nnCostFunction();
		
		
	}

}
