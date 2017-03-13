package deeplearning.test;
import java.io.IOException;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.IntervalRange;

import deeplearning.io.*;
import deeplearning.minimizers.*;
import deeplearning.util.MatrixUtil;
import static org.jblas.MatrixFunctions.*;
import static org.jblas.DoubleMatrix.*;
public class Test {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		int mb = 1024*1024;
    
   /* //Getting the runtime reference from system
    Runtime runtime = Runtime.getRuntime();
     
    System.out.println("##### Heap utilization statistics [MB] #####");
     
    //Print used memory
    System.out.println("Used Memory:"
        + (runtime.totalMemory() - runtime.freeMemory()) / mb);

    //Print free memory
    System.out.println("Free Memory:"
        + runtime.freeMemory() / mb);
     
    //Print total available memory
    System.out.println("Total Memory:" + runtime.totalMemory() / mb);

    //Print Maximum available memory
    System.out.println("Max Memory:" + runtime.maxMemory() / mb);*/
		
		DoubleMatrix n = DoubleMatrix.ones(2,2);
		
		n.mulRow(0,(double)1/3);
		
		MatrixUtil.printMatrix(n);
		
	}

}
