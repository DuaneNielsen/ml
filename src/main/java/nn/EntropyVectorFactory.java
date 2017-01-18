package nn;

import theory.JointEnsembleFactory;

public class EntropyVectorFactory {

	public static EntropyVector uniformVector (int inputLength, int outputLength, int number_samples)  {
		
		int rows = (int)Math.pow(2,inputLength);
		int columns = (int)Math.pow(2,outputLength);
		
		return new EntropyVector(JointEnsembleFactory.uniform(rows, columns), number_samples);
	}
	
}
