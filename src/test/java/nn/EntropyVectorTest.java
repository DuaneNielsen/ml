package nn;

import static org.junit.Assert.*;

import java.util.Iterator;
import java.util.List;

import org.apache.log4j.Logger;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.allocator.enums.Aggressiveness;
import org.nd4j.jita.allocator.enums.AllocationStatus;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.jita.conf.Configuration.AllocationModel;
import org.nd4j.jita.conf.Configuration.ExecutionModel;
import org.nd4j.jita.conf.Configuration.MemoryModel;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import theory.JointEnsembleFactory;

public class EntropyVectorTest {

	Logger log = Logger.getLogger(this.getClass());

	@Before
	public void setUp() throws Exception {
	}

	@Test
	public void testEntropyVector() {
		DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
		
		Nd4j.create(1);
		CudaEnvironment.getInstance().getConfiguration().enableDebug(true).setVerbose(true);
		Nd4j.create(1);
		Configuration config = CudaEnvironment.getInstance().getConfiguration();
	    config.setMaximumDeviceCacheableLength(1024 * 1024 * 1024L)
	    .setMaximumDeviceCache(4L * 1024 * 1024 * 1024L)
	    .setMaximumHostCacheableLength(1024 * 1024 * 1024L)
	    .setMaximumHostCache(4L * 1024 * 1024 * 1024L);
	    
		System.out.println(config);	    
	    
		int num_trials = 1000;
		int num_features = 4;
		int num_label_classes = 2;
		EntropyVector ev = new EntropyVector(JointEnsembleFactory.uniform(num_features, num_label_classes), num_trials);
		DataSet ds = ev.getDataSet();

		Vector input = new Vector(ds.getFeatures());
		Vector output = new Vector(ds.getLabels());

		Iterator<org.nd4j.linalg.dataset.DataSet> iter = ds.iterator();
		
		
		int prob_0_0 = 0;
		int prob_0_1 = 0;
		int prob_3_0 = 0;
		int prob_3_1 = 0;
		
		int rows = 0;
		while (iter.hasNext()) {
			DataSet element = iter.next();
			INDArray features = element.getFeatures();
			INDArray labels = element.getLabels();
			
			System.out.println(features + " " + labels);
			
			if (features.getFloat(0) == 0.0f && features.getFloat(1) == 0.0f && labels.getFloat(0) == 0.0f ) prob_0_0++;
			if (features.getFloat(0) == 0.0f && features.getFloat(1) == 0.0f && labels.getFloat(0) == 1.0f ) prob_0_1++;
			if (features.getFloat(0) == 1.0f && features.getFloat(1) == 1.0f && labels.getFloat(0) == 0.0f ) { prob_3_0++;  };
			if (features.getFloat(0) == 1.0f && features.getFloat(1) == 1.0f && labels.getFloat(0) == 1.0f ) prob_3_1++;
			
			rows++;
		}
		
		System.out.println(rows);
		
		double uniprob = 1.0/ (double)(num_features * num_label_classes);
		
		assertEquals(num_trials,rows);
		assertEquals(uniprob, (double)prob_0_0/(double)num_trials,0.02);
		assertEquals(uniprob, (double)prob_0_1/(double)num_trials,0.02);
		assertEquals(uniprob, (double)prob_3_0/(double)num_trials,0.02);
		assertEquals(uniprob, (double)prob_3_1/(double)num_trials,0.02);
		
	}

//	@Test
//	public void testOutputGeneration() {
//
//		for (int i = 0; i < 100; i++) {
//			EntropyVector ev = new EntropyVector(JointEnsembleFactory.uniform(4, 2),20);
//			DataSet ds = ev.generateInputOutPutPair();
//			Vector input = new Vector(ds.getFeatures());
//			Vector output = new Vector(ds.getLabels());
//			Vector confirmOutput = ev.outputFunction(input);
//			assertTrue(output.getValues().equals(confirmOutput.getValues()));
//		}
//	}
//	
//	@Test
//	public void testVector() {
//
//			EntropyVector ev = new EntropyVector();
//			ev.generateData();
//			
//			DataSet ds = ev.generateInputOutPutPair();
//			
//			Vector input = new Vector(ds.getFeatures());
//			Vector output = new Vector(ds.getLabels());
//			Vector confirmOutput = ev.outputFunction(input);
//			assertTrue(output.getValues().equals(confirmOutput.getValues()));
//
//	}
}
