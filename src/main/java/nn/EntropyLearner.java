package nn;

import java.io.IOException;
import java.util.Iterator;

import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import theory.JointEnsembleFactory;

public class EntropyLearner {
	private static Logger log = LoggerFactory.getLogger(EntropyLearner.class);

	private static UIServer uiServer;
	private static boolean ui = false;

	public static void main(String[] args) throws Exception {

		// logging
		// BasicConfigurator.configure();

		// PLEASE NOTE: For CUDA FP16 precision support is available
		DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);

		// temp workaround for backend initialization
		Nd4j.create(1);

		CudaEnvironment.getInstance().getConfiguration()
				// key option enabled
				.allowMultiGPU(true)

				// we're allowing larger memory caches
				.setMaximumDeviceCache(4L * 1024L * 1024L * 1024L)

				// cross-device access is used for faster model averaging over
				// pcie
				.allowCrossDeviceAccess(true);

		// Initialize the user interface backend
		if (ui) {
			uiServer = UIServer.getInstance();
		}
		new EntropyLearner().run();

	}

	

	private void run() throws IOException {

		// number of rows and columns in the input pictures
		final int inputVectorLength = 4;
		final int outputVectorLength = 2;
		final int number_samples = 200;

		int batchSize = 4; // batch size for each epoch
		int rngSeed = 123; // random number seed for reproducibility
		int numEpochs = 300; // number of epochs to perform
		double rate = 0.1; // learning rate

		// DataSet data = buildDataSet();
		DataSet data = EntropyVectorFactory.uniformVector(inputVectorLength, outputVectorLength, number_samples).getDataSet();
		SplitTestAndTrain split = data.splitTestAndTrain(0.95);
		

		MultiLayerNetwork model = buildModel(inputVectorLength, outputVectorLength, rngSeed, rate);

		model.init();
		
		
		// model.setListeners(new ScoreIterationListener(5)); //print the score
		// with every iteration
		// Configure where the network information (gradients, activations,
		// score vs. time etc) is to be stored
		// Then add the StatsListener to collect this information from the
		// network, as it trains
		StatsStorage statsStorage = new InMemoryStatsStorage(); // Alternative:
																// new
																// FileStatsStorage(File)
																// - see
																// UIStorageExample
		int listenerFrequency = 1;
		model.setListeners(new StatsListener(statsStorage, listenerFrequency));
		// Attach the StatsStorage instance to the UI: this allows the contents
		// of the StatsStorage to be visualized
		if (ui) {
			uiServer.attach(statsStorage);
		}

		model.fit(split.getTrain());

		log.info("Evaluate model....");
		Evaluation eval = new Evaluation(outputVectorLength); //create an evaluation
		//object with 10 possible classes
		Iterator<DataSet> dataIt = split.getTest().iterator();
		 while(dataIt.hasNext()){
		 DataSet next = dataIt.next();
		 INDArray output = model.output(next.getFeatureMatrix()); //get the
		 //networks prediction
		 eval.eval(next.getLabels(), output); //check the prediction against
		 //the true class
		 }

		log.info(eval.stats());
		log.info("****************Example finished********************");

	}

	private MultiLayerNetwork buildModel(final int inputVectorLength,int outputVectorLength, int rngSeed,
			double rate) {
		log.info("Build model....");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(rngSeed) 
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.iterations(2000).miniBatch(false).activation(Activation.RELU).weightInit(WeightInit.XAVIER)
				.learningRate(rate) // specify the learning rate
				.updater(Updater.NESTEROVS).momentum(0.98)//rate of change of learning
				.regularization(true).l2(rate * 0.005) // regularize learning model
				.list()
				.layer(0,
						new DenseLayer.Builder()
								.nIn(inputVectorLength).nOut(4).activation(Activation.SIGMOID).build())
				.layer(1,
						new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // create
																					// hidden
																					// layer
								.activation(Activation.SOFTMAX).nIn(4).nOut(outputVectorLength).build())
				.pretrain(false).backprop(true) // use backpropagation to adjust
												// weights
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);

		return model;
	}

	public DataSet buildDataSet() {

		MyDataVector myData = new MyDataVector(new float[] { 0f, 0f }, new float[] { 0f, 1f });
		myData.addInputOutputPair(new float[] { 0f, 1f }, new float[] { 1f, 0f });
		myData.addInputOutputPair(new float[] { 1f, 0f }, new float[] { 1f, 0f });
		myData.addInputOutputPair(new float[] { 1f, 1f }, new float[] { 0f, 1f });
		return myData.getDataSet();

	}

}
