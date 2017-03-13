package deeplearning.util;

import org.jblas.DoubleMatrix;

public class Parameters {
	private DoubleMatrix W[];
	private DoubleMatrix b[];

	public Parameters(DoubleMatrix W[], DoubleMatrix b[]) {
		this.W = W;
		this.b = b;
	}

	public DoubleMatrix[] getW() {
		return W;
	}

	public DoubleMatrix[] getB() {
		return b;
	}

	public void setW(DoubleMatrix[] W) {
		this.W = W;
	}

	public void setB(DoubleMatrix[] b) {
		this.b = b;
	}

}
