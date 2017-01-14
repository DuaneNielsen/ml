package nn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class VectorFactory {

	/**
	 * creates a Vector in the form (1.0, 0.0, 0.0 .... length)
	 * 
	 * @param length - the length of the Vector
	 * @param index - the value to set to 1.0
	 * @return
	 */
	public static Vector singleValueVector(int length, int column) {
		INDArray singlevalue = Nd4j.zeros(length);
		singlevalue.put(0, column, 1.0f);
		return new Vector(singlevalue);
	}
	
}
