package nn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class MyDataVector {
	
	private INDArray input;
	private INDArray output;
	private int inputlength;
	private int outputlength;
	
	Logger log = LoggerFactory.getLogger(this.getClass());
	
	public MyDataVector(float[] initialInput, float[] initialOutput) {
		super();
		init(initialInput.length, initialOutput.length, initialInput, initialOutput);
	}
	
	public MyDataVector(int inputlength, int outputlength, float[] initialInput, float[] initialOutput) {
		super();
		init(inputlength, outputlength, initialInput, initialOutput);
	}

	private void init(int inputlength, int outputlength, float[] initialInput, float[] initialOutput) {
		this.inputlength = inputlength;
		this.outputlength = outputlength;
		// create empty matrix with no columns
		this.input = Nd4j.create(initialInput, new int[] {1,this.inputlength});
		this.output = Nd4j.create(initialOutput, new int[] {1,this.outputlength});
	}

	public void addInputOutputPair(float[] input, float[] output) {
		
		INDArray inputVector = Nd4j.create(input, new int[]{1,this.inputlength}, 'c');
		INDArray outputVector = Nd4j.create(output, new int[]{1,this.outputlength}, 'c');
		
		this.input  = Nd4j.vstack(this.input, inputVector);
		this.output  = Nd4j.vstack(this.output, outputVector);
		
	}

	public INDArray getInput() {
		return input;
	}

	public INDArray getOutput() {
		return output;
	}

	public int getInputlength() {
		return inputlength;
	}

	public int getOutputlength() {
		return outputlength;
	}
	
	public DataSet getDataSet() {
		return new DataSet(this.input, this.output);
	}
	
	
}
