package nn;

import theory.Ensemble;
import theory.IJointEnsemble;
import theory.JointEnsembleFactory;
import theory.NaiveJointEnsembleGenerator;
import theory.NotAProbabilityDistribution;
import theory.Symbol;
import theory.SymbolPair;

import java.util.*;

import org.nd4j.linalg.dataset.DataSet;

public class EntropyVector extends MyDataVector {

	protected IJointEnsemble<Integer,Integer> joint;
	protected NaiveJointEnsembleGenerator<Integer,Integer> gen;
	private int num_datapoints = 20;
	
	public EntropyVector(IJointEnsemble<Integer,Integer> joint, int num_datapoints ) {
		
		this.joint = joint;
		this.num_datapoints = num_datapoints;
		gen = new NaiveJointEnsembleGenerator<Integer,Integer>(new Random(), new Random(), joint);
		SymbolPair<Integer, Integer> pair = gen.generateRandom();
		float input[] = activationVectorFromInt(pair.row.getSymbol(), joint.rowLength());
		float output[] = activationVectorFromInt(pair.column.getSymbol(), joint.columnLength());
		super.init(binlog(joint.rowLength()), binlog(joint.columnLength()), input, output);
		generateData();
	}
		
	public int getInputLength() {
		return binlog(joint.rowLength());
	}
	
	public int getOutputlength() {
		return binlog(joint.columnLength());
	}
	
	private void generateData() {
		for ( int i = 1; i < num_datapoints; i++) {
			SymbolPair<Integer, Integer> pair = gen.generateRandom();
			float input[] = activationVectorFromInt(pair.row.getSymbol(), joint.rowLength());
			float output[] = activationVectorFromInt(pair.column.getSymbol(), joint.columnLength());
			super.addInputOutputPair(input,output);
		}
	}

	/**
	 * takes an integer, and returns an array of floats, where there is
	 * a "1.0" in each position in the array for the corresponding
	 * bit in the integer that is set to 1
	 * 
	 * @param symbol
	 * @param significantBits
	 * @return
	 */
	public float[] activationVectorFromInt(int symbol, int alphabet_length) {
		int significantBits = binlog(alphabet_length);
		float[] vector = new float[significantBits];
	    for (int i = significantBits -1; i >= 0; i--) {
	    	// convert the integer bits to an array of floats
	        vector[i] = ((symbol & (1 << i)) != 0) ? 1.0f : 0.0f;
	    }
		return vector;
	}
	
	/**
	 * returns the binary log of an integer in bits
	 * 
	 * @param bits
	 * @return
	 */
	public static int binlog( int bits ) // returns 0 for bits=0
	{
	    int log = 0;
	    if( ( bits & 0xffff0000 ) != 0 ) { bits >>>= 16; log = 16; }
	    if( bits >= 256 ) { bits >>>= 8; log += 8; }
	    if( bits >= 16  ) { bits >>>= 4; log += 4; }
	    if( bits >= 4   ) { bits >>>= 2; log += 2; }
	    return log + ( bits >>> 1 );
	}
	
//	public DataSet generateInputOutPutPair() {		
//		Symbol<Vector> input = joint.generateRandom();
//		Vector output = outputFunction(input.getSymbol());
//		return new DataSet(input.getSymbol().getValues(), output.getValues());
//	}
//	
//	public Vector outputFunction(Vector input) {
//		return input;
//	}

}
