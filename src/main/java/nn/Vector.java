package nn;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Vector implements Comparable<Vector> {

	private INDArray values;
	
	public Vector(INDArray vector) {
		this.values = vector;
	}

	public int compareTo(Vector vector) {
		return values.sumNumber().intValue() - vector.values.sumNumber().intValue();
	}

	public INDArray getValues() {
		return values;
	}

	public void setValues(INDArray values) {
		this.values = values;
	}
	
	public int length() {
		return this.values.columns();
	}
	
	public String toString() {
		return values.toString();
	}
	
}
