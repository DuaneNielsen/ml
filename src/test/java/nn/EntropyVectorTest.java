package nn;

import static org.junit.Assert.*;

import org.apache.log4j.Logger;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.DataSet;

public class EntropyVectorTest {

	Logger log = Logger.getLogger(this.getClass());

	@Before
	public void setUp() throws Exception {
	}

	@Test
	public void testEntropyVector() {
		EntropyVector ev = new EntropyVector();
		DataSet ds = ev.getDataSet();

		Vector input = new Vector(ds.getFeatures());
		Vector output = new Vector(ds.getLabels());
		Vector confirmOutput = ev.outputFunction(input);

		assertTrue(output.getValues().equals(confirmOutput.getValues()));
	}

	@Test
	public void testOutputGeneration() {

		for (int i = 0; i < 100; i++) {
			EntropyVector ev = new EntropyVector();
			DataSet ds = ev.generateInputOutPutPair();
			Vector input = new Vector(ds.getFeatures());
			Vector output = new Vector(ds.getLabels());
			Vector confirmOutput = ev.outputFunction(input);
			assertTrue(output.getValues().equals(confirmOutput.getValues()));
		}
	}
	@Test
	public void testVector() {

			EntropyVector ev = new EntropyVector();
			ev.generateData();
			
			DataSet ds = ev.generateInputOutPutPair();
			
			Vector input = new Vector(ds.getFeatures());
			Vector output = new Vector(ds.getLabels());
			Vector confirmOutput = ev.outputFunction(input);
			assertTrue(output.getValues().equals(confirmOutput.getValues()));

	}
}
