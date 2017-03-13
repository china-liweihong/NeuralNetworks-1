package deeplearning.io;

import java.io.IOException;

import com.jmatio.types.*;
import com.jmatio.io.*;

import org.jblas.*;
import java.util.*;

/*
 * Loader class is used for data processing
 * 
 * Currently it just has one function to read a matFile and return an array of matrices
 * 
 * Written By: Haider Adeel Agha
 */

public class Loader {

	public static DoubleMatrix[] loadMatrix (String matFile,int count) {
		
		DoubleMatrix [] matrices = new DoubleMatrix[count];
		MatFileReader mfr = null;

		try {
			mfr = new MatFileReader(matFile);
		} catch (IOException e) {
			e.printStackTrace();
		}

		Map <String,MLArray> content = mfr.getContent();
		Collection <MLArray> col = content.values();
		Object mlarrays[] = col.toArray();
		
		for (int i =0; i <count; i++) {
			MLDouble ml = (MLDouble) mlarrays[i]; 
			double [][] matArr = ml.getArray();
			matrices[i] = new DoubleMatrix(matArr);
		}
		return matrices;
	}

}

