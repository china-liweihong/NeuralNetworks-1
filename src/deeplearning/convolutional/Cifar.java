package deeplearning.convolutional;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.IntervalRange;

import deeplearning.display.Display;
import deeplearning.util.Convolution;
import deeplearning.util.MatrixUtil;
import deeplearning.util.CostGradient;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.lang.Thread;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

import javax.imageio.ImageIO;

public class Cifar {

	
	public DoubleMatrix bd;
	public DoubleMatrix Wd;
	public DoubleMatrix [][][] Wc;
	public DoubleMatrix [] bc;

	private DoubleMatrix Wd_grad;
	private DoubleMatrix [][][] Wc_grad;
	private DoubleMatrix bd_grad;
	private DoubleMatrix [] bc_grad;
		
	private int imageDim;
	private int numClasses;
	private int [] filterDim;
	private int [] numOfFilters;
	private int[] poolDim;
	private int [] inputDim;
	private int [] channels;
	
	private int [] convDim; 
	private int [] outputPool;
	private int [] padded;
	
	private int hiddenSize;
	
	private int numImages;
	private double lambda;
	
	private DoubleMatrix theta;
	private DoubleMatrix errorSoftmax;
	public DoubleMatrix errorsPooled;
	public DoubleMatrix [][][] activations;
	
	private DoubleMatrix [][] images;
	private DoubleMatrix Labels;
	
	public DoubleMatrix nnParams;
	private DoubleMatrix[][][] indexPool;
	
	private String datafile;

	public int layers;
	
	public Cifar(DoubleMatrix [][] images, DoubleMatrix Labels, DoubleMatrix nnParams, int imageDim, int numClasses, int [] filterDim, int [] numOfFilters,
			int [] poolDim, int [] channels, int layers, int [] padded, String datafile, int hiddenSize) {

		this.layers = layers;
		this.datafile = datafile;
		this.imageDim = imageDim;
		this.numClasses = numClasses;
		this.filterDim = filterDim;
		this.numOfFilters = numOfFilters;
		this.lambda = 1e-4;
		
		this.poolDim = poolDim;
		
		this.inputDim = new int[layers];
		this.convDim = new int[layers];
		this.outputPool = new int[layers];
		this.indexPool = new DoubleMatrix[this.layers][][];
		
		this.inputDim[0] = this.imageDim;
		this.convDim[0] = (this.inputDim[0] - this.filterDim[0]) + 1;
		this.outputPool[0] = this.convDim[0]/this.poolDim[0];
		
		
		for ( int i = 1 ; i < this.layers; i++)	{
			
			this.inputDim[i] = this.outputPool[i-1];
			this.convDim[i] = ( this.inputDim[i] - this.filterDim[i]) + 1;
			this.outputPool[i] = this.convDim[i]/this.poolDim[i];
		}
		
		this.hiddenSize = hiddenSize;
		this.Labels = Labels;
		this.images = images;	
		
		this.activations = new DoubleMatrix[this.layers][][];
		
		this.channels = channels;
		
		this.padded = padded;
	}
	
	
	public CostGradient Cost (DoubleMatrix [][] images , DoubleMatrix Labels, DoubleMatrix nnParams)	{

		this.numImages = images.length;
 		InitWeights(nnParams);
		
 		int nonlin = 1;			//0 for sigmoid, 1 for relu
		int Pooling = 1;		//0 for average Pooling, 1 for maxPooling
		
		DoubleMatrix [][][] Pooled = new DoubleMatrix[this.layers][][];
		
		activations[0] = Cifar.Convolution(images, Wc[0], bc[0], 0, nonlin);
		if(Pooling == 1)
			Pooled[0] = this.PoolMax(activations[0], this.poolDim[0], 0);
		else if(Pooling == 0)
			Pooled[0] = this.Pooling(activations[0], 0, this.poolDim[0]);
		
		for( int i = 1 ; i < layers ; i++)	{
			
			activations[i] = Cifar.Convolution(Pooled[i-1], Wc[i], bc[i], i, nonlin);
			if(Pooling == 1)
				Pooled[i] = this.PoolMax(activations[i], this.poolDim[i], i);
			else if(Pooling == 0)
				Pooled[i] = this.Pooling(activations[i], i, this.poolDim[i]);
		}
				
		DoubleMatrix [][][] errorsProp = new DoubleMatrix[this.layers+1][][];
		DoubleMatrix cost  = new DoubleMatrix();
		
		DoubleMatrix activationsReshaped = Cifar.FourDtotwoD(Pooled[layers-1]);	
		cost = computecost(Wd, activationsReshaped, Labels);
		DoubleMatrix errorsPooled = this.errorFc();
		DoubleMatrix [][] errorsReshaped = TwoDtoFourD(errorsPooled,numImages, numOfFilters[this.layers-1], outputPool[this.layers-1], outputPool[this.layers-1]);
		errorsProp[this.layers] = errorsReshaped;
			
		
		for( int i = layers-1 ; i > 0; i--){
			errorsProp[i] = Upsampling(errorsProp[i+1], Pooled[i-1], i, nonlin);
		}
		Upsampling(errorsProp[1], images, 0, nonlin);

		
		DoubleMatrix GradientsRolled = rollWeights(Wc_grad, Wd_grad, bc_grad, bd_grad);
		
		CostGradient ret = new CostGradient();
		ret.setGradient(GradientsRolled);
		ret.setCost(cost.get(0,0));

		return ret;
	}	

	public DoubleMatrix deserializeDM( double [] arr, int rows, int cols)
	{
		
		assert(rows * cols == arr.length) : "error in deserialization";
		
		DoubleMatrix mat = new DoubleMatrix();
		
		for ( int i = 0 ; i < rows ; i++)	{
			
			double [] row = new double[cols];
			
			System.arraycopy(arr, i*cols, row, 0, cols);
			mat.putRow(i, new DoubleMatrix(row));
		}
		return new DoubleMatrix();
	}
	
	public double[] serializeDM( DoubleMatrix mat)
	{
		double [][] matrix = mat.toArray2();
		double [] single = new double[matrix.length*matrix[0].length];
		int arrnum = 0;
		for(double [] arr : matrix){
			System.arraycopy(arr, 0, single, arrnum*matrix[0].length,matrix[0].length);
			arrnum++;
		}
		
		assert( single.length == matrix.length * matrix[0].length) : "errors in serialization";
		
		return single;
	}
	
	
	public int [] predict(DoubleMatrix[][] images , DoubleMatrix nnParams)
	{
		this.numImages = images.length;
		InitWeights(nnParams);

 		int nonlin = 1;			//0 for sigmoid, 1 for relu
		int Pooling = 1;		//0 for average Pooling, 1 for maxPooling
		
		activations[0] = this.Convolution(images, Wc[0], bc[0], 0, nonlin);
		DoubleMatrix [][] Pooled = this.PoolMax(activations[0], poolDim[0], 0);

		activations[1] = this.Convolution(Pooled, Wc[1], bc[1], 1, nonlin);
		DoubleMatrix[][] Pooled2 = this.PoolMax(activations[1], poolDim[1], 1);
		
		activations[2] = this.Convolution(Pooled2, Wc[2], bc[2], 2, nonlin);
		DoubleMatrix[][] Pooled3 = this.PoolMax(activations[2], poolDim[2], 2);
		
		DoubleMatrix activationsReshaped = this.FourDtotwoD(Pooled3);

		int[] maxindex;
		theta = Wd;
		theta=theta.reshape(numClasses,hiddenSize);
		
		DoubleMatrix M=theta.mmul(activationsReshaped);
		
		DoubleMatrix bias = bd.repmat(1, numImages);
		
		//Adding bias
		M = M.add(bias);
		
		M=MatrixFunctions.exp(M);
		
		DoubleMatrix sum= M.columnSums();		
		
		sum=sum.repmat(numClasses,1);
		
		M=M.div(sum);
		
		maxindex=M.columnArgmaxs();
		
		return maxindex;
	}
	
	
	public DoubleMatrix [][] Upsampling(DoubleMatrix [][] PooledErrors, DoubleMatrix [][] images, int lay, int nonlin)
	{
		DoubleMatrix [][] Upsampled = this.UpsampleMaxPooling(PooledErrors, this.poolDim[lay], lay);
		this.calcGradients(lay, Upsampled, images);
		if(lay!=0){
			DoubleMatrix [][] errors = this.errorsPropogation(lay, Upsampled);
			return errors;
		}
		else
			return new DoubleMatrix[1][1];
	}
	
		
	public static DoubleMatrix[][] Convolution(DoubleMatrix [][] input , DoubleMatrix [][] Wc, DoubleMatrix bc, int lay, int nonlin)
	{
		int numImages = input.length;
		int numOfFilters = Wc.length;
		int channels = input[0].length;
		int inputDim = input[0][0].columns;
		int filterDim = Wc[0][0].columns;
		int convDim = inputDim - filterDim + 1;
		//Layer 1-> Convolution Layer
		
 		DoubleMatrix [][] activations = new DoubleMatrix[numImages][numOfFilters];
		
		//System.out.println(input[0][0].get(0,0));
		for ( int i = 0 ; i < numImages ; i++ )	{
			
			//Extracting each image from array of images
			
			double imagetemp[][][] = new double[channels][inputDim][inputDim];
			for(int j = 0 ; j < channels ; j++)	{
				imagetemp[j] = new double[inputDim][inputDim];
				
				imagetemp[j] = input[i][j].toArray2();
			}
			
			
			for( int j = 0 ; j < numOfFilters; j++ )	{
				
				activations[i][j] = DoubleMatrix.zeros(convDim, convDim);
				
				DoubleMatrix [] filter = new DoubleMatrix[channels];
				
				//extracting each filter
				//filter = Wc[j];
				
				double [][][] filt = new double[channels][filterDim][filterDim];
				for(int k = 0 ; k < channels ; k++){
					filt[k] = Wc[j][k].toArray2();
				}
				//convoluting with every image with each image
				
				double [][][] sum = new double[channels][convDim][convDim];
				for( int k = 0 ; k < channels ; k++){
					sum[k]= Convolution.convolution2D(imagetemp[k], inputDim, inputDim, filt[k], filterDim, filterDim);
				}
				double [][] convOut = new double[convDim][convDim];
				for(int k = 0 ; k < channels ; k++)	{
					for(int l = 0 ; l < convDim ; l++)	{
						for(int p = 0 ; p < convDim ; p++)	{
							
							convOut[l][p] = sum[k][l][p] + convOut[l][p]; 
						}
					}
				}
				//System.out.println(Arrays.deepToString(convOut));
				
				DoubleMatrix convolved = new DoubleMatrix(convOut);
				
				//Adding bias
				convolved = convolved.add(bc.get(j));
				
				if(nonlin == 0){
				//Applying the sigmoid non-linearity
				convolved = MatrixFunctions.exp(convolved.neg()).add(1).rdiv(1);
				}
				else if(nonlin == 1){
				//Applying the relu
				convolved = convolved.max(DoubleMatrix.zeros(convolved.rows, convolved.columns));
				}
				//Storing in 4d structure of activations
				activations[i][j] = convolved;
				
				//activations[i][j].print();	
			}
		}			
		
		return activations;
	}

	public DoubleMatrix [][] PoolMax (DoubleMatrix [][] activations, int poolDim, int lay)
	{
		
		int numImages = activations.length;
		int numOfFilters = activations[0].length;
		int dim = activations[0][0].rows;
		int pooled = dim/poolDim;
		
		DoubleMatrix[][] activationsPooled = new DoubleMatrix[numImages][numOfFilters];
		indexPool[lay] = new DoubleMatrix[numImages][numOfFilters];
		//int q_tot = this.poolDim*this.poolDim;
		
		DoubleMatrix avg_ker = DoubleMatrix.zeros(pooled, pooled); 
		for( int i = 0; i < numImages ; i++)	{
			
			for ( int j = 0 ; j < numOfFilters; j++ )	{
				
				activationsPooled[i][j] = new DoubleMatrix(pooled, pooled);
				indexPool[lay][i][j] = DoubleMatrix.zeros(dim, dim);
				
				for ( int k = 0 ; k < pooled; k++)	{
				
					for ( int l = 0 ; l < pooled ; l++)	{
						
						DoubleMatrix temp = (activations[i][j]).getRange((k*poolDim), (k*poolDim)+poolDim, (l*poolDim), (l*poolDim)+poolDim);
						double max = temp.max();
						int ind = temp.argmax();
						
						int cs = ind/poolDim;
						int rs = ind%poolDim;
						indexPool[lay][i][j].put( rs+ (k*poolDim), cs + (l*poolDim), 1);
						activationsPooled[i][j].put( k, l, max);
						
					}
				}
			}
		}
		return activationsPooled;
		
	}
	
	public DoubleMatrix [][] Pooling (DoubleMatrix [][] activations, int lay, int poolDim)
	{
		DoubleMatrix[][] activationsPooled = new DoubleMatrix[numImages][numOfFilters[lay]];
		
		int q_tot = poolDim*poolDim;
		
		DoubleMatrix avg_ker = DoubleMatrix.ones(poolDim, poolDim).div(q_tot);
		
		for( int i = 0; i < this.numImages ; i++)	{
			
			for ( int j = 0 ; j < this.numOfFilters[lay] ; j++ )	{
				
				DoubleMatrix featMap = activations[i][j];
				
				double[][] curFeatMap = featMap.toArray2();
				double[][] avgker = avg_ker.toArray2();
				
				//convoluting to replace activations with average over poolingRegion
				double convOut[][] = Convolution.convolution2D(curFeatMap, convDim[lay], convDim[lay], avgker, poolDim, poolDim);

				//downsampling convOut to get 1 value from region of PoolDim*poolDim
				double[][] downsampled = MatrixUtil.downsample(convOut , convDim[lay] , poolDim);
				
				//Storing in 4d structure of activationsPooled
				activationsPooled[i][j] = new DoubleMatrix(downsampled);
				
			}
		}
		return activationsPooled;

	}

	
	public static DoubleMatrix FourDtotwoD (DoubleMatrix [][] input)
	{
		//unrolling activationsPooled
		
		int fourthD = input.length;
		int thirdD = input[0].length;
		int secD = input[0][0].getRows();
		int firstD = input[0][0].getColumns();
		
		//For converting 4d of (10x10x20xnumImages) to 3d of (10x200xnumImages)
		DoubleMatrix[] activations3d = new DoubleMatrix[fourthD];
		
		//For converting 3d of : 10 x 200 x numImages ---> 2d of : 2000 x numImages
		//1d array is not appropriate, but it is converted to DoubleMatrix
		DoubleMatrix[] activations2d = new DoubleMatrix[fourthD];
		
		int hiddenSize = firstD * secD * thirdD;
		
		//hiddenSize = outPool*outPool*numFilters --> 2000 in this case
		DoubleMatrix activationsReshaped = new DoubleMatrix(hiddenSize, fourthD);
		
		for(int i = 0 ; i < fourthD ; i++ )	{
			
			activations3d[i] = input[i][0];
			for( int j = 1; j < thirdD ; j++)	{
				
				activations3d[i] = DoubleMatrix.concatHorizontally(activations3d[i], input[i][j]);
			}
			
			activations2d[i] = activations3d[i].getColumn(0);
			for( int j = 1; j < secD * thirdD ; j++ )	{
				
				activations2d[i] = DoubleMatrix.concatVertically(activations2d[i], activations3d[i].getColumn(j));
			}
			
			activationsReshaped.putColumn(i, activations2d[i]);
		}
		
		assert( activationsReshaped.columns* activationsReshaped.rows == input.length*
				input[0].length * input[0][0].rows* input[0][0].columns) : "error in 4d to two d";
		
		return activationsReshaped;
	}

	
	
	public DoubleMatrix errorFc()
	{
		//Error of Pooling layer = Weights of Softmax * error of softmax layer
		DoubleMatrix errorsPooled = Wd.transpose().mmul(errorSoftmax);
		
		return errorsPooled = errorsPooled.transpose();
		
	}
	
	public DoubleMatrix [][] TwoDtoFourD(DoubleMatrix input, int fourthD, int thirdD, int secondD, int firstD)
	{
		
		assert( input.rows * input.columns == fourthD*thirdD*secondD*firstD) : "error in Two D to FourD";
		
		int lay = layers - 1;
		//RESHAPINIG OUR POOLING LAYER ERRORS OF 2000 X numOfImages ---> 10 x 10 x 20 x numImages
		DoubleMatrix[][] errorsReshaped= new DoubleMatrix[fourthD][thirdD];
		DoubleMatrix[] singleFilter = new DoubleMatrix[thirdD]; 

		int image=0;
		for ( int i = 0 ; i < fourthD ; i++ )	{
			
			DoubleMatrix imageerror = input.getRow(i);
			
			for (int j =0 ; j < thirdD ; j++)	{
				
				int x = j*(secondD * firstD) , y =  x + (firstD * firstD ) ;
				
				singleFilter[j] = imageerror.getRange(x, y);
				singleFilter[j].reshape( firstD , firstD );
				errorsReshaped[i][j] = singleFilter[j];
			}
			//image=+ ( outputPool * outputPool * numOfFilters);
		}

		return errorsReshaped;
		
	}
	
	public DoubleMatrix[][] UpsampleMaxPooling( DoubleMatrix [][] PooledErrors, int poolDim, int lay)
	{
		int outputDim = PooledErrors[0][0].rows*poolDim;
		DoubleMatrix[][] errorUpsampled = new DoubleMatrix[numImages][numOfFilters[lay]];
		
		for ( int i = 0 ; i < numImages ; i++)	{	
			for ( int j = 0 ; j < PooledErrors[0].length ; j++)	{
				
				errorUpsampled[i][j] = new DoubleMatrix(convDim[lay], convDim[lay]);
				for ( int k = 0 ; k < outputPool[lay] ; k++)	{
					
					for ( int l = 0 ; l < outputPool[lay] ; l++)	{
						
						//DoubleMatrix aux = DoubleMatrix.ones(poolDim, poolDim);
						IntervalRange rr = new IntervalRange(k*poolDim, (k*poolDim)+poolDim);
						IntervalRange cr = new IntervalRange(l*poolDim, (l*poolDim)+poolDim);
						DoubleMatrix aux = indexPool[lay][i][j].get(rr,cr);
						aux = aux.mul(PooledErrors[i][j].get(k,l));
						
						errorUpsampled[i][j].put(rr, cr, aux);
					}
				}
				DoubleMatrix aux3 = errorUpsampled[i][j];
				
				//Taking gradient of the relu activation function
				errorUpsampled[i][j] = aux3.mul(activations[lay][i][j].truth());
			}
		}
		return errorUpsampled;
	}

	public void calcGradients( int lay, DoubleMatrix [][] errorUpsampled, DoubleMatrix [][] images )
	{
		for( int i = 0 ; i < numImages; i++){
			
			DoubleMatrix [] cur_img = images[i];
			
			for(int j = 0 ; j < numOfFilters[lay] ; j++)	{
				
				for( int k = 0 ; k < channels[lay] ; k++)	{
				
					//multiplying respective error units with the image to get Gradients of Convolution layer
					double[][] noww = Convolution.convolution2D(cur_img[k].toArray2(), inputDim[lay], inputDim[lay], errorUpsampled[i][j].toArray2(), convDim[lay], convDim[lay]);
					DoubleMatrix nowww = new DoubleMatrix(noww);
				
					//Aggregating the gradients
					Wc_grad[lay][j][k] = Wc_grad[lay][j][k].add(nowww);
				}
				
				bc_grad[lay].put(j, bc_grad[lay].get(j) + errorUpsampled[i][j].sum());
			}
		}
	}

	
	DoubleMatrix [][] errorsPropogation( int lay, DoubleMatrix [][] errorUpsampled )
	{
		
		DoubleMatrix[][] errorsProp= new DoubleMatrix[numImages][numOfFilters[lay-1]];
		
		for( int i = 0 ; i < numImages; i++){
			
			DoubleMatrix [] cur_err= errorUpsampled[i];
			
			for(int j = 0 ; j < numOfFilters[lay-1]; j++)	{
				errorsProp[i][j] = new DoubleMatrix(outputPool[lay-1], outputPool[lay-1]);
			}
			
			for(int j = 0 ; j < errorUpsampled[0].length ; j++)	{				
				
				DoubleMatrix error = cur_err[j];
				
				for( int k = 0 ; k < channels[lay] ; k++)	{
				
					//multiplying respective error units with the image to get Gradients of Convolution layer
					DoubleMatrix padd = padding(error, padded[lay], padded[lay]);
					DoubleMatrix filter = spatialflip(Wc[lay][j][k]);
//					DoubleMatrix filter = Wc[lay][j][k];
					double[][] noww = Convolution.convolution2D(padd.toArray2(), padd.columns, padd.rows, filter.toArray2(), filter.columns, filter.rows);
					DoubleMatrix nowww = new DoubleMatrix(noww);
				
					//Aggregating the gradients
					errorsProp[i][k] = errorsProp[i][k].add(nowww);
				}
			}
		}
		return errorsProp;
	}
		

	public DoubleMatrix squarederror(DoubleMatrix theta, DoubleMatrix data, DoubleMatrix labels)
	{
		int inputSize = this.hiddenSize;
		theta = theta.reshape(numClasses,inputSize);
		DoubleMatrix res = theta.mmul(data);
		
		
		return labels;	
	}
	
	public DoubleMatrix fc(DoubleMatrix Theta, DoubleMatrix bias, DoubleMatrix input, int out)
	{
		//DoubleMatrix in = this.FourDtotwoD(input);
		
		int inputSize = input.rows;
		
		Theta = Theta.reshape(out, inputSize);
		
		DoubleMatrix output = Theta.mmul(input);
		
		bias = bias.repmat(1, inputSize);
		
		output = output.add(bias);
		
		return output;
	}
	
	public DoubleMatrix computecost(DoubleMatrix theta, DoubleMatrix data,DoubleMatrix labels)
	{
		int numClasses = this.numClasses;
		int inputSize = this.hiddenSize;
		double lambda = this.lambda; 
		
		int numCases=this.numImages;
		
		double negdivide=-1*numCases;	

		theta=theta.reshape(numClasses,inputSize);
		
		DoubleMatrix M=theta.mmul(data);
		
		DoubleMatrix bias = bd.repmat(1, numCases);
		
		//Adding bias
		M = M.add(bias);

		DoubleMatrix groundTruth=DoubleMatrix.zeros(numClasses,numCases);
		
		for(int x=0;x<numCases;x++)
		{			
			int temp=(int) labels.get(x,0);
			
			//PUT -1 when using CSV
			if(datafile == "mnist")
				groundTruth.put(temp - 1,x,1.0);
			else if(datafile == "cifar")
				groundTruth.put(temp,x,1.0);
			
		}
		
		DoubleMatrix maximums=M.columnMaxs();
	

		maximums=maximums.repmat(numClasses,1);

		M=M.sub(maximums);

		DoubleMatrix h=MatrixFunctions.exp(M);

		DoubleMatrix sum= h.columnSums();
		sum=sum.repmat(numClasses,1);
		h=h.div(sum);
	
		//System.out.println("PRINTING H ___________");
		
		DoubleMatrix firstcost=(((groundTruth.mul(MatrixFunctions.log(h))).columnSums()).rowSums()).div(negdivide);
		
		DoubleMatrix secondcost=((MatrixFunctions.pow(theta,2).columnSums()).rowSums()).mul(lambda/2);
//		DoubleMatrix secondcost = DoubleMatrix.zeros(firstcost.rows, firstcost.columns);
		
		DoubleMatrix costreg=firstcost.add(secondcost);
		//DoubleMatrix costreg = firstcost; 
		
		//HAIDERS ADDITION
		this.errorSoftmax = groundTruth.sub(h).div(negdivide);
		//System.out.println("_______ERROR LENGHT : " + error_2.rows + " "  + error_2.columns);
		this.Wd_grad=(((groundTruth.sub(h)).mmul(data.transpose())).div(negdivide)).add(theta.mmul(lambda));
		//this.Wd_grad=(((groundTruth.sub(h)).mmul(data.transpose())).div(negdivide));
		
		
		//Wd_grad=Wd_grad.reshape(inputSize*numClasses,1);
		bd_grad = this.errorSoftmax.mmul(DoubleMatrix.ones(numImages,1));
		
		return costreg;

	}
	
	//Salman's Kron
	public static double[][] kron(double[][] a, double[][] b)
	{
		int p=a.length;
		int q=a[0].length;
		int m=b.length;
		int n=b[0].length;
				
		
		int i,j;
		double[][] c=new double[p*m][q*n];
		
		for(int x=0;x<p;x++)
		{
			for(int y=0;y<q;y++)
			{
				
				//for each element of a we traverse through the matrix b
				for(int x1=0;x1<m;x1++)
				{
					for(int y1=0;y1<n;y1++)
					{
						i=(m*((x+1)-1))+(x1+1);
						j=(n*((y+1)-1)+(y1+1));
						
						c[i-1][j-1]=a[x][y]*b[x1][y1];
						
								
					}	
				}
			}
		}
		
		return c;
	}

	
	public void unrollWeights(DoubleMatrix nnParams)	{
		
		int sum = 0 ;
		
		for ( int lay = 0 ; lay < this.layers ; lay++){
			
			for( int i = 0 ; i < numOfFilters[lay] ; i++)	{
				
				IntervalRange rr = new IntervalRange(i*(filterDim[lay]*filterDim[lay]*channels[lay]) + sum, (i+1)*(filterDim[lay]*filterDim[lay]*channels[lay]) + sum);
				DoubleMatrix temp = nnParams.get(rr, 0);
				
				for ( int j = 0 ; j < channels[lay] ; j++)	{
				
					Wc[lay][i][j] = temp.getRange(j*filterDim[lay]*filterDim[lay], (j+1)*filterDim[lay]*filterDim[lay]).reshape(filterDim[lay], filterDim[lay]);			
				}
			}			
			sum += filterDim[lay]*filterDim[lay]*channels[lay]*numOfFilters[lay];			
		}

		IntervalRange wdRange = new IntervalRange(sum,sum + hiddenSize*numClasses);
		Wd = nnParams.get(wdRange , 0);
		
		int bias = 0;
		for (int i = 0 ; i < this.layers ; i++)	{
			
			bias = bias + numOfFilters[i];
			IntervalRange bcRange = new IntervalRange(sum+(hiddenSize*numClasses),sum+(hiddenSize*numClasses) +  numOfFilters[i]);
			bc[i] = nnParams.get(bcRange , 0);
		}
		
		IntervalRange bdRange = new IntervalRange(sum + (hiddenSize*numClasses) + bias, (sum + (hiddenSize*numClasses) + bias) + numClasses);
		
		bd = nnParams.get( bdRange , 0);
	}


	
	
	public DoubleMatrix  rollWeights(DoubleMatrix [][][] Wc, DoubleMatrix Wd, DoubleMatrix [] bc, DoubleMatrix bd)
	{
		//unrolling Filters
		//assumes only square filters
		
		int layers = Wc.length;
		int [] filters = new int[layers];
		int [] channels = new int[layers];
		int [] filterDim = new int[layers];
		
		for(int i = 0  ;i < layers ; i++)	{
			filters[i] = Wc[i].length;
			channels[i] = Wc[i][0].length;
			filterDim[i] = Wc[i][0][0].columns;
		}
		
		DoubleMatrix [] temp = new DoubleMatrix[this.layers];
		
		for ( int lay = 0 ; lay < layers ; lay++)	{
			
			DoubleMatrix [] tempo = new DoubleMatrix[filters[lay]];

			for( int i = 0 ; i < filters[lay] ; i++)	{
				
				tempo[i] = Wc[lay][i][0].reshape(filterDim[lay]*filterDim[lay], 1);
				for ( int j = 1 ; j < channels[lay]; j++)	{
					
					tempo[i] = DoubleMatrix.concatVertically(tempo[i], Wc[lay][i][j].reshape(filterDim[lay]*filterDim[lay], 1));
				}
			}
			temp[lay] = tempo[0];
			for ( int i = 1 ; i < filters[lay] ; i++)	{
				temp[lay] = DoubleMatrix.concatVertically(temp[lay], tempo[i]);
			}
		}
		
		DoubleMatrix nnParams = temp[0];
		for( int i = 1 ; i < layers; i++)
			nnParams = DoubleMatrix.concatVertically(nnParams, temp[i]);
			
		Wd = Wd.reshape(Wd.rows * Wd.columns, 1);
		nnParams = DoubleMatrix.concatVertically(nnParams, Wd);	

		DoubleMatrix bias = bc[0];
		for( int i = 1 ; i < layers ; i++)	
			bias = DoubleMatrix.concatVertically(bias, bc[i]);
		
		nnParams = DoubleMatrix.concatVertically(nnParams, bias);
		nnParams = DoubleMatrix.concatVertically(nnParams, bd);
		
		//this.nnParams = temp;
		return nnParams;
	}
	
	public void DisplayWeights(DoubleMatrix [][] Wc)	
	{
		int numOfFilters = Wc.length;
		int dimension = Wc[0][0].rows;
		
		double [][][][] image = new double[numOfFilters][3][dimension][dimension];
		for ( int i = 0 ; i < numOfFilters ; i++)	{
			for ( int j = 0 ; j < 3 ; j++)	{
				image[i][j] = Wc[i][j].toArray2();
			}
		}
		try {
			Display.DisplayMatrixImg(image, numOfFilters);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	public void DisplayActivations(DoubleMatrix [][] Wc, int n)	
	{
		int numOfFilters = Wc.length;
		int dimension = Wc[0][0].rows;
		
		double [][][][] image = new double[numOfFilters][1][dimension][dimension];

		for ( int j = 0 ; j < numOfFilters ; j++)	{
			image[0][j] = Wc[0][j].toArray2();
		}

		try {
			Display.DisplayMatrixImg(image, 1);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	
	
	public void InitWeights(DoubleMatrix params)
	{
		
		Wd = new DoubleMatrix(hiddenSize, this.numClasses);
		Wc = new DoubleMatrix[this.layers][][];
		for( int i = 0 ; i < this.layers ; i++)	{
			Wc[i] = new DoubleMatrix[this.numOfFilters[i]][this.channels[i]];
		}
		
		for( int lay = 0 ; lay < this.layers ; lay++)	{
			for(int i = 0; i < numOfFilters[lay]; i++)	{
				for( int j = 0 ; j < channels[lay]; j++)	{
					
					Wc[lay][i][j] = new DoubleMatrix(filterDim[lay], filterDim[lay]);
				}
			}
		}			

		bd = DoubleMatrix.zeros(this.numClasses, 1);
		
		bc = new DoubleMatrix[this.layers];
		for ( int i = 0 ; i < this.layers ; i++)	{
			bc[i] = DoubleMatrix.zeros(numOfFilters[i], 1);
		}
		
		Wc_grad = new DoubleMatrix[this.layers][][];
		for ( int i = 0 ; i < this.layers ; i++)	{	
			Wc_grad[i] = new DoubleMatrix[numOfFilters[i]][channels[i]];
		}
		
		Wd_grad = new DoubleMatrix(Wd.rows , Wd.columns);
		
		for( int lay = 0 ; lay < this.layers ; lay++)	{
			for ( int i = 0 ; i < numOfFilters[lay] ; i++)	{
	 			for(int j = 0 ; j < channels[lay] ; j++)	{
	 				
					Wc_grad[lay][i][j] = new DoubleMatrix(Wc[lay][i][j].rows,Wc[lay][i][j].columns);
	 			}
			}
		}		
		bc_grad = new DoubleMatrix[this.layers];
		for ( int i = 0 ; i < this.layers ; i++)	{
			bc_grad[i] = new DoubleMatrix(bc[i].rows , bc[i].columns);
		}
 		bd_grad = new DoubleMatrix(bd.rows , bd.columns);
 					
		unrollWeights(params);
 		
	}

	public static DoubleMatrix spatialflip(DoubleMatrix filter)
	{
		
		DoubleMatrix flipped = new DoubleMatrix(filter.rows, filter.columns);
		
		for ( int i = 0 ; i < filter.rows ; i++)	{
			for ( int j = 0 ; j < filter.columns ; j++)	{
				
				flipped.put(filter.rows - i - 1, filter.columns - j - 1, filter.get(i,j));
			}
		}
		return flipped;
	}

	
	public static DoubleMatrix padding(DoubleMatrix input, int rows, int columns)	{

		DoubleMatrix out = DoubleMatrix.concatHorizontally(DoubleMatrix.zeros(input.rows, columns), input);
		out = DoubleMatrix.concatHorizontally(out, DoubleMatrix.zeros(out.rows, columns));
		out = DoubleMatrix.concatVertically(DoubleMatrix.zeros(rows, out.columns), out);
		out = DoubleMatrix.concatVertically(out, DoubleMatrix.zeros(rows, out.columns));
		
		return out;
	}

	public DoubleMatrix[][] BackpropInASingleFunc(DoubleMatrix [][] PooledErrors, DoubleMatrix [][] images, int lay, int nonlin)
	{
		//POOLING LAYER ERRORS = 10 x 10 x 20 x numImages

		//For Upsampling of errors to 20 x 20 x 20 x numImages
		DoubleMatrix[][] errorUpsampled = new DoubleMatrix[numImages][numOfFilters[lay]];
		
		for ( int i = 0 ; i < numImages ; i++)	{
			
			for ( int j = 0 ; j < numOfFilters[lay] ; j++)	{
				
				errorUpsampled[i][j] = new DoubleMatrix(convDim[lay], convDim[lay]);
				
				for ( int k = 0 ; k < outputPool[lay] ; k++)	{
					
					for ( int l = 0 ; l < outputPool[lay] ; l++)	{
						
						//DoubleMatrix aux = DoubleMatrix.ones(poolDim, poolDim);
						IntervalRange rr = new IntervalRange(k*poolDim[lay], (k*poolDim[lay])+poolDim[lay]);
						IntervalRange cr = new IntervalRange(l*poolDim[lay], (l*poolDim[lay])+poolDim[lay]);
						DoubleMatrix aux = indexPool[lay][i][j].get(rr,cr);
						aux = aux.mul(PooledErrors[i][j].get(k,l));
						
						errorUpsampled[i][j].put(rr, cr, aux);
					}
				}
				DoubleMatrix aux3 = errorUpsampled[i][j];
				
				//KRON takes 2 matrices of A of size m by n and B of size p by q and multiplies each 
				//element of A by the matrix B. 
				//RESULTS SIZE = m*p by n*q
				
				//-------FOR AVERAGE POOLING ------//
//				double [][] upsampled = new double[convDim[lay]][convDim[lay]];
//				upsampled = kron(PooledErrors[i][j].toArray2() , DoubleMatrix.ones(poolDim,poolDim).toArray2());
//				errorUpsampled[i][j] = new DoubleMatrix(upsampled);
//				double divi = ((double)1/(double)(poolDim*poolDim));
//				DoubleMatrix aux3 = errorUpsampled[i][j].mul(divi); 
				
				//Dividing each error unit by average - to offset the multiplicative effect of
				//convolution later
				
				if(nonlin == 0){
				//Taking gradient of the sigmoid activation function
					errorUpsampled[i][j] = aux3.mul(activations[lay][i][j]).mul(activations[lay][i][j].neg().add(1));
				}
				else if (nonlin == 1){
				//Taking gradient of the relu activation function
					errorUpsampled[i][j] = aux3.mul(activations[lay][i][j].truth());
				}
			}
		}
		
		
		for( int i = 0 ; i < numImages; i++){
			
			DoubleMatrix [] cur_img = images[i];
			
			for(int j = 0 ; j < numOfFilters[lay] ; j++)	{
				
				for( int k = 0 ; k < channels[lay] ; k++)	{
				
					//multiplying respective error units with the image to get Gradients of Convolution layer
					double[][] noww = Convolution.convolution2D(cur_img[k].toArray2(), inputDim[lay], inputDim[lay], errorUpsampled[i][j].toArray2(), convDim[lay], convDim[lay]);
					DoubleMatrix nowww = new DoubleMatrix(noww);
				
					//Aggregating the gradients
					Wc_grad[lay][j][k] = Wc_grad[lay][j][k].add(nowww);
				}
				
				bc_grad[lay].put(j, bc_grad[lay].get(j) + errorUpsampled[i][j].sum());
			}
		}
		
		if( lay != 0)	{
			
			DoubleMatrix[][] errorsProp= new DoubleMatrix[numImages][numOfFilters[lay-1]];
			
			for( int i = 0 ; i < numImages; i++){
				
				DoubleMatrix [] cur_err= errorUpsampled[i];
				
				for(int j = 0 ; j < numOfFilters[lay-1]; j++)	{
					errorsProp[i][j] = new DoubleMatrix(outputPool[lay-1], outputPool[lay-1]);
				}
				
				for(int j = 0 ; j < errorUpsampled[0].length ; j++)	{				
					
					DoubleMatrix error = cur_err[j];
					
					for( int k = 0 ; k < channels[lay] ; k++)	{
					
						//multiplying respective error units with the image to get Gradients of Convolution layer
						DoubleMatrix padd = padding(error, padded[lay], padded[lay]);
						DoubleMatrix filter = spatialflip(Wc[lay][j][k]);
//						DoubleMatrix filter = Wc[lay][j][k];
						double[][] noww = Convolution.convolution2D(padd.toArray2(), padd.columns, padd.rows, filter.toArray2(), filter.columns, filter.rows);
						DoubleMatrix nowww = new DoubleMatrix(noww);
					
						//Aggregating the gradients
						errorsProp[i][k] = errorsProp[i][k].add(nowww);
					}
				}
			}
			return errorsProp;
		}
		
		DoubleMatrix [][] useless = new DoubleMatrix[1][1];
		return useless;
	}

}
