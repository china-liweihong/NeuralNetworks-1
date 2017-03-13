package deeplearning.convolutional;

import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import javax.imageio.ImageIO;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.IntervalRange;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLUInt8;
import com.opencsv.CSVReader;
//import com.sun.javafx.collections.MappingChange.Map;

import deeplearning.util.CostGradient;
import deeplearning.util.MatrixUtil;
import deeplearning.convolutional.Cifar;
import deeplearning.display.Display;

//TODO : MAKE IN OOP ; ROLLING AND UNROLLING MECHANISM ; LAYERS ADDITION

public class Convolutional {
	
	private static int imageDim;
	private static int numClasses;
	private static int [] filterDim;
	private static int [] inputDim;
	private static int [] numOfFilters;
	private static int [] poolDim;
	private static int [] convDim; 
	private static int [] outputPool;
	private static int [] padding;
	private static int [] channels;
	private static int hiddenSize;
	
	public static DoubleMatrix nnParams;
	public static int layers;
	public static String datafile;
	
	public int numImages;
	DoubleMatrix images[];
	DoubleMatrix[][] colImages;
	DoubleMatrix [][] pictures;
	DoubleMatrix Labels;
	
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
		//specifying the type of data 
		datafile = "cifar";
		
		Convolutional cn = new Convolutional();
		
		String[] arr = new String[6];
		arr[0] = "data/data_batch_1.mat";
		arr[1] = "data/data_batch_2.mat";
		arr[2] = "data/data_batch_3.mat";
		arr[3] = "data/data_batch_4.mat";
		arr[4] = "data/data_batch_5.mat";
		arr[5] = "data/test_batch.mat";

		cn.numImages = 20000;
		cn.colImages = new DoubleMatrix[cn.numImages][channels[0]];
		cn.Labels = DoubleMatrix.zeros(cn.numImages, 1);
		System.out.println(cn.numImages/10000 );
		if (datafile == "mnist"){
			cn.loadCSV("data/mnist_train.csv", cn.numImages);
		}
		else if(datafile == "cifar"){
			for( int i = 0 ; i < cn.numImages/10000 ; i++){
				DoubleMatrix [][] images = cn.loadCIFAR( 10000 ,arr[i],i);
				System.arraycopy(images,0,cn.colImages,(i)*10000,10000);
			}
		}
		
		cn.initParameters();
		
		cn.CifarStochastic();
		
		Convolutional.nnParams.save("Parameters.txt");		
		
		cn.numImages = 1000;
		cn.colImages = new DoubleMatrix[cn.numImages][Convolutional.channels[0]];
		cn.Labels = new DoubleMatrix(cn.numImages,1);
		
		if(datafile == "cifar"){
			cn.colImages = cn.loadCIFAR(cn.numImages, arr[5],0);
		}
		else if (datafile == "mnist")	{
			cn.loadCSV("data/mnist_test.csv", cn.numImages);
		}

		int[] pred = cn.predict(cn.colImages, new DoubleMatrix("Parameters.txt"));
		
//		displayImages(colImages, 2);
//		costing.DisplayWeights(costing.Wc[0]);
		//costing.DisplayWeights(costing.activations[0],5);
//		costing.DisplayWeights(costing.Wc[1]);
		
	 	double count=0.0;
	 	if(datafile == "mnist"){
		 	for(int x=0;x<pred.length;x++)
		 	{
		 		if((pred[x] + 1 )==(int)cn.Labels.get(x,0))
		 		{
		 			count++;
		 			//System.out.println(Labels.get(x,0));
		 		}
		 	}
	 	}
	 	else if(datafile == "cifar")
	 	{
	 		for(int x=0;x<pred.length;x++)
		 	{
		 		if((pred[x])==(int)cn.Labels.get(x,0))
		 		{
		 			count++;
		 			//System.out.println(Labels.get(x,0));
		 		}
		 	}
	 	}

	 	double size=(double) pred.length;
	 	double accuracy=(count/size)*100;
	 			
	 	System.out.println("Accuracy is :" +accuracy);

	}

	public Convolutional() {
		
		Convolutional.layers = 3;
		
		if(datafile == "mnist"){
			Convolutional.imageDim = 28;
		}
		else if(datafile == "cifar"){
			Convolutional.imageDim = 32;
		}
		
		Convolutional.numClasses = 10;
		
		Convolutional.padding = new int[layers];
		padding[0] = 0;
		padding[1] = 3;
		padding[2] = 2;
		
		Convolutional.inputDim = new int[layers];
		Convolutional.filterDim = new int[layers];
		Convolutional.numOfFilters = new int[layers];
		Convolutional.channels = new int[layers];
		Convolutional.convDim = new int[layers];
		Convolutional.outputPool = new int[layers];
		
		Convolutional.poolDim = new int[layers];
		
		Convolutional.inputDim[0] = imageDim;
		Convolutional.filterDim[0] = 3;
		Convolutional.numOfFilters[0] = 32;
		Convolutional.poolDim[0] = 2;
		Convolutional.channels[0] = channels[0];
		Convolutional.convDim[0] = (Convolutional.inputDim[0] - Convolutional.filterDim[0]) + 1;
		Convolutional.outputPool[0] = Convolutional.convDim[0]/Convolutional.poolDim[0];
		
		Convolutional.inputDim[1] = Convolutional.outputPool[0];
		Convolutional.filterDim[1] = 4;
		Convolutional.numOfFilters[1] = 32;
		Convolutional.poolDim[1] = 2;
		Convolutional.channels[1] = Convolutional.numOfFilters[0];
		Convolutional.convDim[1] = (Convolutional.inputDim[1] - Convolutional.filterDim[1]) + 1;
		Convolutional.outputPool[1] = Convolutional.convDim[1]/Convolutional.poolDim[1];
		
		Convolutional.inputDim[2] = Convolutional.outputPool[1];
		Convolutional.filterDim[2] = 3;
		Convolutional.numOfFilters[2] = 64;
		Convolutional.poolDim[2] = 1;
		Convolutional.channels[2] = Convolutional.numOfFilters[1];
		Convolutional.convDim[2] = (Convolutional.inputDim[2] - Convolutional.filterDim[2]) + 1;
		Convolutional.outputPool[2] = Convolutional.convDim[2]/Convolutional.poolDim[2];
		
		Convolutional.hiddenSize = Convolutional.outputPool[layers-1] * Convolutional.outputPool[layers-1] * Convolutional.numOfFilters[layers-1];
		
		//this.outputPool = 2;
		 //this.lambda = 1e-4;
	}
	

	public Convolutional(int imageDim, int numClasses, int [] filterDim, int [] numOfFilters,
						int [] poolDim, int inputchannel, int layers) {
		
		Convolutional.layers = layers;
		Convolutional.imageDim = imageDim;
		Convolutional.numClasses = numClasses;
		Convolutional.filterDim = filterDim;
		Convolutional.numOfFilters = numOfFilters;
		Convolutional.poolDim = poolDim;

		Convolutional.channels = new int[layers]; Convolutional.channels[0] = inputchannel;
		for(int i = 1 ; i < layers ; i++){
			Convolutional.channels[i] = Convolutional.numOfFilters[i-1];
		}
		
		Convolutional.convDim = new int[layers];
		for(int i = 0 ; i < layers ; i++)
			Convolutional.convDim[i] = (Convolutional.inputDim[i] - Convolutional.filterDim[i]) + 1;
		
		Convolutional.outputPool = new int[layers];
		for ( int i = 0 ; i < layers; i++)	{
			Convolutional.outputPool[i] = Convolutional.convDim[i]/Convolutional.poolDim[i];
		}
		
		Convolutional.inputDim = new int[layers]; Convolutional.inputDim[0] = imageDim;
		for( int i = 1 ; i < layers ; i++)	{
			Convolutional.inputDim[i] = Convolutional.outputPool[i-1];
		}
		
		Convolutional.padding = new int[layers]; Convolutional.padding[0] = 0;
		for(int i = 1 ; i < layers ; i++){
			Convolutional.padding[i] = Convolutional.filterDim[i]-1;
		}
		hiddenSize = Convolutional.outputPool[layers-1] * Convolutional.outputPool[layers-1] * Convolutional.numOfFilters[layers-1];
	}
	
	
	
	//Runs ConvNetCost on random minibatches from the data and updates gradients
	
	//Runs ConvNetCost on random minibatches from the data and updates gradients
	public synchronized void CifarStochastic()
	{
		int m= Labels.getLength();
		int minibatch=128;
		double momentum=0.9;
		double epochs=1;
		double alpha = .01;
		double mom=0.5;
		double momIncrease=20;
		
		
		DoubleMatrix Vel = new DoubleMatrix(nnParams.rows, nnParams.columns);
		
		int it=0;

		Cifar costing = new Cifar(colImages, Labels, nnParams, imageDim, numClasses, filterDim, numOfFilters,
				poolDim, channels, layers, padding, datafile, hiddenSize);

		for(int e=0; e<epochs ;e++)
		{
			
			Random rand = new Random();
			int [] rp = new int[m];
			
			for( int i =0 ; i < m ; i++ )	{
				rp[i] = rand.nextInt(m);		
			}
			
			for(int s = 1; s < m -(minibatch+1); s+=minibatch)
			{
				it=it+1;
				
				if(it==(int) momIncrease)
				{
					mom=momentum;
				}
				
				DoubleMatrix[][] mb_data = new DoubleMatrix[minibatch][Convolutional.channels[0]];
				DoubleMatrix mb_labels = DoubleMatrix.zeros(minibatch);
				
				for ( int i =0 ; i < minibatch ; i++){
					for(int j = 0 ; j < Convolutional.channels[0] ; j++)	{
						mb_data[i][j] = colImages[ (int) (rp[s + i - 1 ])][j];
						mb_labels.put(i, Labels.get((int)(rp[s + i - 1])));
					}
				}
				
				CostGradient cost_g = new CostGradient();
				
				
				//System.out.println(nnParams);
				
				cost_g = costing.Cost(mb_data, mb_labels, nnParams);
				
				double cost = cost_g.getCost();
				DoubleMatrix grad = cost_g.getGradient();
				
				DoubleMatrix temp = new DoubleMatrix(grad.rows,grad.columns);
				temp = grad.mul(alpha);
				Vel = Vel.mul(mom).add(temp);
				nnParams = nnParams.sub(Vel);
				
				System.out.println("Epoch " +e+": Cost on iteration "+it+" is "+cost);
			}
//			if((e+1)%2==0)
//			alpha = alpha/2.0;
		}
		
	}

	
	@SuppressWarnings("resource")
	public void loadCSV(String dataFile , int dataSize) throws IOException {
		
		CSVReader reader;
		this.numImages = dataSize;
		images = new DoubleMatrix[dataSize];
		Convolutional.channels[0] = 1;
		Convolutional.imageDim = 28;
		
		reader = new CSVReader(new FileReader(dataFile));
	    
		String [] nextLine;
	    //Labels = DoubleMatrix.zeros(dataSize, 1);
	    int imagenum = 0;
	    
	    // nextLine[] is an array of values of the image 
	    while ((nextLine = reader.readNext()) != null && imagenum < dataSize) {
	    	
	    	if(Double.parseDouble(nextLine[0]) != 0)	
	    		Labels.put(imagenum, 0 , Double.parseDouble(nextLine[0]));
	    	
	    	else	
	    		Labels.put(imagenum, 0, 10);
	    	
    		images[imagenum] = DoubleMatrix.zeros(Convolutional.imageDim ,Convolutional.imageDim);
    		
	    	for(int i = 0; i < Convolutional.imageDim ; i++){	    			    		
	    		for(int j = 0; j < Convolutional.imageDim ; j++)	{
	    			
	    			images[imagenum].put( i , j , Double.parseDouble(nextLine[(i*28)+j+1])/255.0);
	    			
	    			//System.out.print(images[imagenum].get(i,j) + " ");
	    		}
	    		//System.out.println();
	    	}
	    	
	    	//System.out.println(Labels.getRow(imagenum));	     
	        //System.out.println(images.getRow(imagenum));
	        //System.out.println(Double.parseDouble(nextLine[300]));
	    	colImages[imagenum] = new DoubleMatrix[Convolutional.channels[0]];
	    	colImages[imagenum][0] = images[imagenum];
	    	imagenum++;
	    }
	}
	
	@SuppressWarnings("resource")
	public void loadtest(int dataSize) throws FileNotFoundException, IOException
	{
		CSVReader reader;
		this.numImages = dataSize;
		this.colImages = new DoubleMatrix[dataSize][3];
		Convolutional.channels[0] = 3;
		Convolutional.imageDim = 28;
		reader = new CSVReader(new FileReader("data/mnist_train.csv"));
	    
		String [] nextLine;
	    Labels = DoubleMatrix.zeros(dataSize, 1);
	    int imagenum = 0;
	    
	    // nextLine[] is an array of values of the image 
	    while ((nextLine = reader.readNext()) != null && imagenum<dataSize) {
	    	
	    	if(Double.parseDouble(nextLine[0]) != 0)	
	    		Labels.put(imagenum, 0 , Double.parseDouble(nextLine[0]));
	    	
	    	else	
	    		Labels.put(imagenum, 0, 10);
	    	
	    	for( int k =0 ; k < channels[0] ; k++){
	    		colImages[imagenum][k] = DoubleMatrix.zeros(Convolutional.imageDim ,Convolutional.imageDim);
    		
		    	for(int i = 0; i < imageDim ; i++){	    			    		
		    		for(int j = 0; j < imageDim ; j++)	{
		    			
		    			colImages[imagenum][k].put( i , j , Double.parseDouble(nextLine[(i*28)+j+1])/255.0);
		    			
		    			//System.out.print(images[imagenum].get(i,j) + " ");
		    		}
		    		//System.out.println();
		    	}
	    	}	
	    	//System.out.println(Labels.getRow(imagenum));	     
	        //System.out.println(images.getRow(imagenum));
	        //System.out.println(Double.parseDouble(nextLine[300]));
	    	
	    	imagenum++;
	    }
	}
	
	public DoubleMatrix[][] loadCIFAR(int dataSize, String filename, int file) throws FileNotFoundException, IOException
	{
		Convolutional.imageDim = 32;
		MatFileReader matfilereader = new MatFileReader(filename);
		//MLDouble mlArr = (MLDouble)matfilereader.getMLArray("data");
		
		java.util.Map<String, MLArray> mymap = matfilereader.getContent();
		MLUInt8 data = (MLUInt8) mymap.get("data");
		MLUInt8 labels = (MLUInt8) mymap.get("labels");
		
		Convolutional.channels[0] = 3;
		//this.numImages = dataSize;
		//this.colImages = new DoubleMatrix[numImages][channels];
		//DoubleMatrix [][] colImages = new DoubleMatrix[numImages][channels];
		

		byte[][] arrayw = labels.getArray();
		
		//DoubleMatrix Labelstemp = new DoubleMatrix(numImages,1);
		
		for( int i = 0 ; i < dataSize; i++)	{
			Labels.put(i+(10000*file), 0, arrayw[i][0]);
		}	
		
		byte[][] arr = data.getArray();
		data.getM();
	data.getN();
		
		DoubleMatrix [][] images = new DoubleMatrix[dataSize][3];
		
		for ( int i = 0 ; i < dataSize ; i++)	{
			
			for ( int j = 0 ; j < 3 ; j++)	{
				 
				images[i][j] = new DoubleMatrix(32,32);
				
					for ( int k = 0 ; k < 32*32 ; k++)	{
					
						if( arr[i][(1024*j) + k] > -1)
							images[i][j].put(k/32, k%32, (arr[i][1024*j+k])/255.0);
						
						else{
							images[i][j].put(k/32, k%32, (arr[i][1024*j+k]+256)/255.0);
						}
						
					}
				}
		}
		
//		for ( int i =0 ; i < 32 ; i++)	{
//			for ( int j = 0 ; j < 32 ; j++)	{
//				
//				System.out.println("i: " + i + "  j: " + j + "  " + images[0][1].get(i, j));
//			}
//		}

		//MatFileReader labels = new MatFileReader("data/data_batch_1.mat");
		
		//this.colImages = images;
		return images;
	}
	
	public void loadData(String dataFile) throws IOException
	{
		
		//DO NOT USE
		//JAVA HEAP OUT OF MEMORY
		DoubleMatrix images = new DoubleMatrix();
		DoubleMatrix.loadCSVFile(dataFile);
		
		System.out.println(images.getRow(0));	
	}
	
	// Initializes encoding and decoding parameters
	public void initParameters() {

		DoubleMatrix [][][] Wc = new DoubleMatrix[Convolutional.layers][][];
		for( int lay = 0 ; lay < layers ; lay++)	{
			
			Wc[lay] = new DoubleMatrix[ Convolutional.numOfFilters[lay]][channels[lay]];
			for(int i=0; i<Convolutional.numOfFilters[lay]; i++)	{
				for(int j = 0; j < channels[lay] ; j++)	{
					// i for each feature map
					//Wd[i] = MatrixUtil.randInitializeConv((this.outputPool*this.outputPool), this.numClasses);
					Wc[lay][i][j] = DoubleMatrix.randn(Convolutional.filterDim[lay], Convolutional.filterDim[lay]);
					Wc[lay][i][j] = Wc[lay][i][j].mul(0.1);
				}
			}
		}
		
		DoubleMatrix Wd = new DoubleMatrix();
		Wd = MatrixUtil.randInitializeConv(hiddenSize, Convolutional.numClasses);
		DoubleMatrix bd = DoubleMatrix.zeros(Convolutional.numClasses, 1);	
		
		DoubleMatrix [] bc = new DoubleMatrix[Convolutional.layers];
		for ( int lay = 0 ; lay < layers ; lay++)	{
			bc[lay] = DoubleMatrix.zeros(Convolutional.numOfFilters[lay], 1);
		}
		// bd = 10 x 1
		int totalParams = 0;
		
		for ( int lay = 0 ; lay < layers ; lay++)	{
			totalParams = totalParams + filterDim[lay]*filterDim[lay]*numOfFilters[lay] + numOfFilters[lay];
		}
		totalParams = totalParams + (hiddenSize*numClasses) + numClasses;
		
		nnParams = new DoubleMatrix(totalParams, 1);
		
		nnParams = rollWeights(Wc, Wd, bc, bd);
	}


	public void initParameters(DoubleMatrix Params) {

		DoubleMatrix [][][] Wc = new DoubleMatrix[Convolutional.layers][][];
		for( int lay = 0 ; lay < layers ; lay++)	{
			
			Wc[lay] = new DoubleMatrix[ Convolutional.numOfFilters[lay]][channels[lay]];		
		}
		
		DoubleMatrix Wd = new DoubleMatrix();
		nnParams = new DoubleMatrix(Params.length, 1);

		DoubleMatrix bd = DoubleMatrix.zeros(Convolutional.numClasses, 1);	
		
		DoubleMatrix [] bc = new DoubleMatrix[Convolutional.layers];
		for ( int lay = 0 ; lay < layers ; lay++)	{
			bc[lay] = DoubleMatrix.zeros(Convolutional.numOfFilters[lay], 1);
		}
		
		unrollWeights(Params, Wc, bc, Wd, bd);
	}

	
	//For matrix unrolling into a single vector
	public DoubleMatrix  rollWeights(DoubleMatrix [][][] Wc, DoubleMatrix Wd, DoubleMatrix [] bc, DoubleMatrix bd)
	{
		
		DoubleMatrix [] temp = new DoubleMatrix[Convolutional.layers];
		
		for ( int lay = 0 ; lay < Convolutional.layers ; lay++)	{
			
			DoubleMatrix [] tempo = new DoubleMatrix[numOfFilters[lay]];

			for( int i = 0 ; i < numOfFilters[lay] ; i++)	{
				
				tempo[i] = Wc[lay][i][0];
				for ( int j = 1 ; j < channels[lay]; j++)	{
					
					tempo[i] = DoubleMatrix.concatHorizontally(tempo[i], Wc[lay][i][j]);
				}
			}
			
			temp[lay] = tempo[0].reshape(filterDim[lay]*filterDim[lay]*channels[lay], 1);
			for(int x=1 ; x < numOfFilters[lay] ; x++)
			{
				temp[lay] = DoubleMatrix.concatVertically(temp[lay], tempo[x].reshape(filterDim[lay]*filterDim[lay]*channels[lay], 1));
			}
		}
		
		DoubleMatrix nnParams = temp[0];
		for( int i = 1 ; i < Convolutional.layers; i++)
			nnParams = DoubleMatrix.concatVertically(nnParams, temp[i]);
			
		nnParams = DoubleMatrix.concatVertically(nnParams, Wd.reshape(hiddenSize*numClasses, 1));	

		DoubleMatrix bias = bc[0];
		for( int i = 1 ; i < Convolutional.layers ; i++)	
			bias = DoubleMatrix.concatVertically(bias, bc[i]);
		
		nnParams = DoubleMatrix.concatVertically(nnParams, bias);
		nnParams = DoubleMatrix.concatVertically(nnParams, bd);
		
		//this.nnParams = temp;
		return nnParams;
	}
	
	
//	public DoubleMatrix  rollGradients()
//	{
//		
//		DoubleMatrix temp=Wc_grad[0].reshape(filterDim*filterDim, 1);
//		for(int x=1 ; x < numOfFilters ; x++)
//		{
//			temp = DoubleMatrix.concatVertically(temp, Wc_grad[x].reshape(filterDim*filterDim, 1));
//		}
//		
//		temp = DoubleMatrix.concatVertically(temp, Wd_grad.reshape(hiddenSize*numClasses, 1));	
//		temp = DoubleMatrix.concatVertically(temp, bc_grad);
//		temp = DoubleMatrix.concatVertically(temp, bd_grad);
//		
//		//this.nnParams = temp;
//		return temp;
//	}
//	
		
	
 		
//	
//	public void UpdateGradients() { 
//		
//		//SUBTRACTING GRADIENTS FROM THE PARAMETERS
//		for( int i = 0 ; i < numOfFilters ; i++)	{
//			
//			Wc[i] = Wc[i].sub(Wc_grad[i]);
//		}
//		
//		bc = bc.sub(bc_grad);
//		Wd = Wd.sub(Wd_grad);
//		
//	}
//	
	//FUNCTION TO GET predictions from our updated weights
	//will be replaced later to get cost from ConvNet Cost function
	
	public double[][] rotateMatrixLeft(double[][] matrix)
	
	{
		//http://stackoverflow.com/questions/42519/how-do-you-rotate-a-two-dimensional-array
		//Martijn Courteaux
		
	    /* W and H are already swapped */
	    int w = matrix.length;
	    int h = matrix[0].length;   
	    double[][] ret = new double[h][w];
	    for (int i = 0; i < h; ++i) {
	        for (int j = 0; j < w; ++j) {
	            ret[i][j] = matrix[j][h - i - 1];
	        }
	    }
	    return ret;
	}
	
		
	public static void displayImages( DoubleMatrix [][] images , int samples){
		 
		double [][][][] image = new double[samples][3][32][32];
		for ( int i = 0 ; i < samples ; i++)	{
			for ( int j = 0 ; j < 3 ; j++)	{
				image[i][j] = images[i][j].toArray2();
			}
		}
		try {
			Display.DisplayMatrixImg(image, samples);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	public int [] predict(DoubleMatrix[][] images , DoubleMatrix nnParams)
	{
		
		this.numImages = images.length;
		
		DoubleMatrix bd = new DoubleMatrix();
		DoubleMatrix Wd = new DoubleMatrix();
		DoubleMatrix [][][] Wc = new DoubleMatrix[Convolutional.layers][][];
		for( int i = 0 ; i < Wc.length ; i++){
			Wc[i] = new DoubleMatrix[Convolutional.numOfFilters[i]][Convolutional.channels[i]];
		}
		DoubleMatrix [] bc = new DoubleMatrix[Convolutional.layers];
			
		int sum = 0 ;
		
		for ( int lay = 0 ; lay < Convolutional.layers ; lay++){
			
			for( int i = 0 ; i < Convolutional.numOfFilters[lay] ; i++)	{
				
				IntervalRange rr = new IntervalRange(i*(Convolutional.filterDim[lay]*Convolutional.filterDim[lay]*Convolutional.channels[lay]) + sum, (i+1)*(Convolutional.filterDim[lay]*Convolutional.filterDim[lay]*Convolutional.channels[lay]) + sum);
				DoubleMatrix temp = nnParams.get(rr, 0);
				
				for ( int j = 0 ; j < Convolutional.channels[lay] ; j++)	{
				
					Wc[lay][i][j] = temp.getRange(j*Convolutional.filterDim[lay]*Convolutional.filterDim[lay], (j+1)*Convolutional.filterDim[lay]*Convolutional.filterDim[lay]).reshape(Convolutional.filterDim[lay], Convolutional.filterDim[lay]);			
				}
			}			
			sum += Convolutional.filterDim[lay]*Convolutional.filterDim[lay]*Convolutional.channels[lay]*Convolutional.numOfFilters[lay];			
		}

		IntervalRange wdRange = new IntervalRange(sum,sum + Convolutional.hiddenSize*Convolutional.numClasses);
		Wd = nnParams.get(wdRange , 0);
		
		int bias = 0;
		for (int i = 0 ; i < Convolutional.layers ; i++)	{
			
			bias = bias + Convolutional.numOfFilters[i];
			IntervalRange bcRange = new IntervalRange(sum+(hiddenSize*numClasses),sum+(hiddenSize*numClasses) +  numOfFilters[i]);
			bc[i] = nnParams.get(bcRange , 0);
		}
		
		IntervalRange bdRange = new IntervalRange(sum + (hiddenSize*numClasses) + bias, (sum + (hiddenSize*numClasses) + bias) + numClasses);
		
		bd = nnParams.get( bdRange , 0);
		
 		int nonlin = 1;			//0 for sigmoid, 1 for relu
		DoubleMatrix [][] active= Cifar.Convolution(images, Wc[0], bc[0], 0, nonlin);
		active = this.PoolMax(active, poolDim[0], 0);

		active = Cifar.Convolution(active, Wc[1], bc[1], 1, nonlin);
		active = this.PoolMax(active, poolDim[1], 1);
		
		active = Cifar.Convolution(active, Wc[2], bc[2], 2, nonlin);
		active = this.PoolMax(active , poolDim[2], 2);
		
		DoubleMatrix activationsReshaped = Cifar.FourDtotwoD(active);

		int[] maxindex;
		DoubleMatrix theta = Wd;
		theta=theta.reshape(numClasses,hiddenSize);
		
		DoubleMatrix M=theta.mmul(activationsReshaped);
		
		DoubleMatrix biased = bd.repmat(1, numImages);
		
		//Adding bias
		M = M.add(biased);
		
		M=MatrixFunctions.exp(M);
		
		DoubleMatrix sumer = M.columnSums();		
		
		sumer = sumer.repmat(numClasses,1);
		
		M = M.div(sum);
		
		maxindex = M.columnArgmaxs();
		
		return maxindex;
	}

	
	public void unrollWeights(DoubleMatrix nnParams, DoubleMatrix [][][] Wc, DoubleMatrix [] bc,
			DoubleMatrix Wd, DoubleMatrix bd)	{
		
		int sum = 0 ;
		
		for ( int lay = 0 ; lay < Convolutional.layers ; lay++){
			
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
		for (int i = 0 ; i < Convolutional.layers ; i++)	{
			
			bias = bias + numOfFilters[i];
			IntervalRange bcRange = new IntervalRange(sum+(hiddenSize*numClasses),sum+(hiddenSize*numClasses) +  numOfFilters[i]);
			bc[i] = nnParams.get(bcRange , 0);
		}
		
		IntervalRange bdRange = new IntervalRange(sum + (hiddenSize*numClasses) + bias, (sum + (hiddenSize*numClasses) + bias) + numClasses);
		
		bd = nnParams.get( bdRange , 0);
	}


	public DoubleMatrix [][] PoolMax (DoubleMatrix [][] activations, int poolDim, int lay)
	{
		
		int numImages = activations.length;
		int numOfFilters = activations[0].length;
		int dim = activations[0][0].rows;
		int pooled = dim/poolDim;
		
		DoubleMatrix[][] activationsPooled = new DoubleMatrix[numImages][numOfFilters];
		//int q_tot = this.poolDim*this.poolDim;
		
		DoubleMatrix.zeros(pooled, pooled); 
		for( int i = 0; i < numImages ; i++)	{
			
			for ( int j = 0 ; j < numOfFilters; j++ )	{
				
				activationsPooled[i][j] = new DoubleMatrix(pooled, pooled);
				
				for ( int k = 0 ; k < pooled; k++)	{
				
					for ( int l = 0 ; l < pooled ; l++)	{
						
						DoubleMatrix temp = (activations[i][j]).getRange((k*poolDim), (k*poolDim)+poolDim, (l*poolDim), (l*poolDim)+poolDim);
						double max = temp.max();
						activationsPooled[i][j].put( k, l, max);
						
					}
				}
			}
		}
		return activationsPooled;
		
	}

	
	@SuppressWarnings("resource")
	public void ReadFromFile(String file)
	{
		
		try {
			new BufferedReader(new FileReader(file));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	
	public int predictSingleImage(String image) throws IOException
	{
		
		BufferedImage img = ImageIO.read((new File(image)));
		System.out.println("File read");
		DoubleMatrix [][] imageinput = new DoubleMatrix[1][3];
		int w = img.getWidth();
		int h = img.getHeight();
		double[][] pixels = new double[w][h];
		
		Convolutional.channels[0] = 3;
		System.out.println("width : " + w + "height : " + h);
		for( int ch = 0 ; ch < 3 ; ch++){
			for(int i = 0; i < w; i++){
			    for(int j = 0; j < h; j++){
			        pixels[i][j] = img.getRGB(i, j);
			    }
			}
			imageinput[0][ch] = new DoubleMatrix(pixels);
		}
			
		System.out.println("Done with image");
		
		DoubleMatrix Params = new DoubleMatrix("C:/Users/Raza/git/DeepLearning/DeepLearning/Parameters.txt");

		System.out.println("Time to start prediction");
		int [] ret = predict(imageinput, Params);
		return ret[0];
	}

}
