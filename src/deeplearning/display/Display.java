package deeplearning.display;

import org.jblas.*;

/*
 * Display class to display images i.e. edge detectors, features learned etc.
 * 
 * Written By: Muhammad Zaheer & Salman Saeed
 * Dated : 15th November, 2014
 * 
 */


import java.io.IOException;
import java.awt.*;
import java.awt.image.*;
import javax.swing.*;

public class Display {
	
	public void DisplayNetwork(DoubleMatrix W) {
		
		// W : 25 x 64
		
		DoubleMatrix images = W.dup();
		
		int m = W.rows;
		
		// Norm bounding the parameters
		for (int i = 0; i < m; i++) {
			DoubleMatrix squaredMatrix = W.getRow(i).mmul(W.getRow(i).transpose());
			
			squaredMatrix = MatrixFunctions.sqrt(squaredMatrix);
			
			images.mulRow(i, ((double) 1) / squaredMatrix.get(0, 0));
		}
		
		int sz = (int)Math.sqrt(W.columns);
		double[][][] imagesNew = new double[W.rows][sz][sz];
		double max = -1;
		double min = 1;
		for (int i = 0; i < 25; i++) 
			for (int j = 0; j < 8; j++)
				for (int k = 0; k < 8; k++) {
					double val = images.get(i,(j*8)+k);
					
					if(val>max)
						max = val;
					if(val<min)
						min = val;
					
					imagesNew[i][j][k] = val;
		}
		
		// Normalizing and scaling to 0-255
		for (int i = 0; i < 25; i++) {
			for (int j = 0; j < 8; j++)
				for (int k = 0; k < 8; k++) {
					double val = images.get(i,(j*8)+k);
					
					val = (imagesNew[i][j][k] - min)*( 1 / (max - min)) ;
							
					imagesNew[i][j][k] = (double)(int)(val * 255);
				}
		}
		
		try {
			//DisplayMatrixImg(imagesNew,25);
		}
		catch (Exception e) {
			
		}
		
	}

	public static void DisplayMatrixImg(double[][][][] image, int samples)
			throws IOException {
		JFrame editorFrame = new JFrame("GrayScale in Matrix/Array Shown as Image");

		editorFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		int xpos = 0;
		int ypos = 0;
		int h = image[0][0][0].length;
		int w = image[0][0].length;

		JLabel[] jlabel = new JLabel[image.length];
		for (int x = 0; x < samples; x++) {
			double imageout[] = new double[image[0][0].length * image[0][0][0].length];
			int px = 0;
			
			for (int i = 0; i < h; i++) {
				for (int j = 0; j < w; j++) {
					imageout[((i * h) + j)] = (int)(image[x][0][i][j]*255.0);
					//imageout[((i * 32) + j)*3 + cha] = 255;
					
				}
			}

			BufferedImage imagebuffer = new BufferedImage(w, h,
					BufferedImage.TYPE_INT_RGB);

			WritableRaster r = (WritableRaster) imagebuffer.getData();
			r.setPixels(0, 0, w, h, imageout);
			imagebuffer.setData(r);

			BufferedImage img1 = imagebuffer;
			BufferedImage img = resize(img1, 100, 100);

			ImageIcon imageIcon = new ImageIcon(img);
			jlabel[x] = new JLabel();
			jlabel[x].setBounds(xpos, ypos, 100, 100);
			jlabel[x].setIcon(imageIcon);
			editorFrame.add(jlabel[x]);
			xpos = xpos + 100;
			if (xpos == 1300) {
				xpos = 0;
				ypos = ypos + 100;
			}

		}

		editorFrame.pack();
		editorFrame.setLocationRelativeTo(null);
		editorFrame.setSize(new Dimension(1325, 650));
		editorFrame.setLocation(10, 50);
		editorFrame.setVisible(true);

	}

	public static BufferedImage resize(BufferedImage image, int width, int height) {
		BufferedImage bi = new BufferedImage(width, height,
				BufferedImage.TRANSLUCENT);
		Graphics2D g2d = (Graphics2D) bi.createGraphics();
		g2d.addRenderingHints(new RenderingHints(RenderingHints.KEY_RENDERING,
				RenderingHints.VALUE_RENDER_QUALITY));
		g2d.drawImage(image, 0, 0, width, height, null);
		g2d.dispose();
		return bi;
	}

}
