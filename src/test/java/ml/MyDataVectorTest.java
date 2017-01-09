package ml;

import java.util.Iterator;

import org.apache.log4j.Logger;
import org.nd4j.linalg.dataset.DataSet;

import junit.framework.TestCase;
import nn.MyDataVector;

public class MyDataVectorTest extends TestCase {

	Logger log = Logger.getLogger(this.getClass());
	
	
	protected void setUp() throws Exception {
		super.setUp();
	}

	public void testMyDataVector() {
		
		MyDataVector data = new MyDataVector(new float[]{0.0f, 0.0f},new float[]{0.0f, 0.0f});
		log.debug(data.getInput());
		assertEquals(0.0, data.getInput().getFloat(0, 0), 0.0);
		assertEquals(0.0, data.getInput().getFloat(0, 1), 0.0);
		
		log.debug(data.getOutput());
		assertEquals(0.0, data.getOutput().getFloat(0, 0), 0.0);
		assertEquals(0.0, data.getOutput().getFloat(0, 1), 0.0);
	}

	public void testAddInputOutputPair() {
		MyDataVector data = new MyDataVector(new float[]{0.0f, 0.0f},new float[]{0.0f, 0.0f});
		
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
	
	public void testDataSet() {
		MyDataVector data = new MyDataVector(new float[]{0.0f, 0.0f},new float[]{0.0f, 0.0f});
		data.addInputOutputPair(new float[]{1.0f,1.0f}, new float[]{1.0f,1.0f});
		
		DataSet dataset = data.getDataSet();
		Iterator<DataSet> i = dataset.iterator();
		while (i.hasNext()) {
			DataSet d = i.next();
			log.debug("feature " + d.getFeatureMatrix());
			log.debug("label" + d.getLabels());
			assertTrue(d.getFeatureMatrix().equals(d.getLabels()));
		}
		
	}

}
