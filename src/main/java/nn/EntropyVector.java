package nn;

import theory.Ensemble;
import theory.NotAProbabilityDistribution;
import theory.Symbol;

import java.util.*;

import org.nd4j.linalg.dataset.DataSet;

public class EntropyVector extends MyDataVector {

	private Ensemble<Vector> inputs;
	private int num_datapoints = 20;
	private int num_inputs = 2;
	
	public EntropyVector() {
		try {
			inputs = new Ensemble<Vector>(new Random(), ensemble());
		} catch (NotAProbabilityDistribution e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		DataSet d = generateInputOutPutPair();
		
		super.init(d.getFeatures(), d.getLabels());
	}
	
	private List<Symbol<Vector>> ensemble() {
		List<Symbol<Vector>> symbols = new ArrayList<Symbol<Vector>>();
		for (int i = 0; i < num_inputs; i++) {
			Vector v = VectorFactory.singleValueVector(num_inputs, i);
			double probability = 1.0/(double)num_inputs;
			Symbol<Vector> s = new Symbol<Vector>(v,probability);
			symbols.add(s);
		}
		return symbols;
	}
	
	public void generateData() {
		for ( int i = 0; i < num_datapoints; i++) {
			Symbol<Vector> input = inputs.generate();
			Vector output = outputFunction(input.getSymbol());
			super.addInputOutputPair(input.getSymbol(), output);
		}
	}

	public DataSet generateInputOutPutPair() {		
		Symbol<Vector> input = inputs.generate();
		Vector output = outputFunction(input.getSymbol());
		return new DataSet(input.getSymbol().getValues(), output.getValues());
	}
	
	public Vector outputFunction(Vector input) {
		return input;
	}

}
