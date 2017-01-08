package ml;

import org.apache.log4j.Logger;
import org.nd4j.linalg.api.ndarray.INDArray;

import junit.framework.TestCase;
import nn.MyDataVector;

public class MyDataVectorTest extends TestCase {

	Logger log = Logger.getLogger(this.getClass());
	
	
	protected void setUp() throws Exception {
		super.setUp();
	}

	public void testMyDataVector() {
		
		MyDataVector data = new MyDataVector(2,2);
		
	}

	public void testAddInputOutputPair() {
		MyDataVector data = new MyDataVector(2,2);
		
		data.addInputOutputPair(new float[]{1.0f,1.0f}, new float[]{1.0f,1.0f});
		
		log.debug(data.getInput());
		assertEquals(0.0, data.getInput().getFloat(0, 0), 0.0);
		assertEquals(0.0, data.getInput().getFloat(0, 1), 0.0);
		

		log.debug(data.getOutput());
		assertEquals(0.0, data.getOutput().getFloat(0, 0), 0.0);
		assertEquals(0.0, data.getOutput().getFloat(0, 1), 0.0);

		log.debug(data.getInput());
		assertEquals(1.0, data.getInput().getFloat(1, 0), 0.0);
		assertEquals(1.0, data.getInput().getFloat(1, 1), 0.0);
		

		log.debug(data.getOutput());
		assertEquals(1.0, data.getOutput().getFloat(1, 0), 0.0);
		assertEquals(1.0, data.getOutput().getFloat(1, 1), 0.0);		
		
	}

}
