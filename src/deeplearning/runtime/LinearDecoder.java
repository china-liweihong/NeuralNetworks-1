package deeplearning.runtime;

import org.jblas.DoubleMatrix;
import deeplearning.display.Display;

import deeplearning.sparseautoencoder.*;
import deeplearning.util.CostGradient;
import deeplearning.util.Parameters;

public class LinearDecoder {

	public static void main (String [] args)	{
		
		double imageChannels = 3;
		double patchDim = 8;
		double numofpatches = 100000;
		double visibleSize = patchDim * patchDim * imageChannels;
		double output = visibleSize;
		double hiddenSize = 400;
		
		double SparsityParam = 0.035;
		double lambda = 0.0003;
		double beta = 5;
		double epsilon = 0.1;
		
		int debugHidden = 5;
		int debugVisible = 3;
		
		SparseAutoencoderLinearCost sac = new SparseAutoencoderLinearCost(debugVisible,debugHidden,lambda,SparsityParam,beta);
		
		DoubleMatrix patchesrand = DoubleMatrix.rand(8, 10);
		
		sac.initParameters();
		
	}
	
}
