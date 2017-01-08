package nn;

import java.io.IOException;
import java.util.Iterator;

import org.apache.log4j.BasicConfigurator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
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
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class EntropyLearner {
    private static Logger log = LoggerFactory.getLogger(EntropyLearner.class);

    public static void main(String[] args) throws Exception {
    	
    	//logging
    	BasicConfigurator.configure();
    	
        // PLEASE NOTE: For CUDA FP16 precision support is available
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);

        // temp workaround for backend initialization
        Nd4j.create(1);

        CudaEnvironment.getInstance().getConfiguration()
            // key option enabled
            .allowMultiGPU(true)

            // we're allowing larger memory caches
            .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)

            // cross-device access is used for faster model averaging over pcie
            .allowCrossDeviceAccess(true);   	
    	
        new EntropyLearner().run();

    }

	private void run() throws IOException {
		
		
		//number of rows and columns in the input pictures
        final int numRows = 2;
        final int numColumns = 1;
        int outputNum = 2; // number of output classes
        
        int batchSize = 128; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 10; // number of epochs to perform
        double rate = 0.0015; // learning rate

        
        DataSet data = buildDataSet();
        
        log.debug(data.toString());
        
        //Get the DataSetIterators:
        //DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        //DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);


        MultiLayerNetwork model = buildModel(numRows, numColumns, outputNum, rngSeed, rate);
        model.init();
        
        model.setListeners(new ScoreIterationListener(5));  //print the score with every iteration

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
        	log.info("Epoch " + i);
            model.fit(data);
        }
        
        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        Iterator<DataSet> dataIt = data.iterator();
        while(dataIt.hasNext()){
            DataSet next = dataIt.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");
        
        log.debug(data.toString());
        
	}

	private MultiLayerNetwork buildModel(final int numRows, final int numColumns, int outputNum, int rngSeed,
			double rate) {
		log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed) //include a random seed for reproducibility
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // use stochastic gradient descent as an optimization algorithm
            .iterations(1)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .learningRate(rate) //specify the learning rate
            .updater(Updater.NESTEROVS).momentum(0.98) //specify the rate of change of the learning rate.
            .regularization(true).l2(rate * 0.005) // regularize learning model
            .list()
            .layer(0, new DenseLayer.Builder() //create the first input layer.
                    .nIn(numRows * numColumns)
                    .nOut(2)
                    .build())
            .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                    .activation(Activation.RELU)
                    .nIn(2)
                    .nOut(outputNum)
                    .build())
            .pretrain(false).backprop(true) //use backpropagation to adjust weights
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
		return model;
	}
	
	

	private DataSet buildDataSet() {
		// list off input values, 4 training samples with data for 2
        // input-neurons each
        INDArray input = Nd4j.zeros(4, 2);

        // correspondending list with expected output values, 4 training samples
        // with data for 2 output-neurons each
        INDArray labels = Nd4j.zeros(4, 2);
        
        NDArrayFactory fac = Nd4j.factory();
        
        //INDArray input = array2x4(new float[][]{{1,0},{0,4}});
        //INDArray labels = array2x4(new float[][]{{1,0},{0,4}});
        
        
        //int [] myarray = ArrayUtil.flatten(new int[][] {{0,0},{0,1}});
        //log.debug(fac.create(new int[]{0,0,0,0}, new int[] {2,2}).toString()); 
        
        
//
//        // create first dataset
//        // when first input=0 and second input=0
//        input.putScalar(new int[]{0, 0}, 0);
//        input.putScalar(new int[]{0, 1}, 1);
//        // then the first output fires for false, and the second is 0 (see class
//        // comment)
//        labels.putScalar(new int[]{0, 0}, 0);
//        labels.putScalar(new int[]{0, 1}, 1);
//
//        // when first input=1 and second input=0
//        input.putScalar(new int[]{1, 0}, 0);
//        input.putScalar(new int[]{1, 1}, 1);
//        // then xor is true, therefore the second output neuron fires
//        labels.putScalar(new int[]{1, 0}, 0);
//        labels.putScalar(new int[]{1, 1}, 1);
//
//        // same as above
//        input.putScalar(new int[]{2, 0}, 0);
//        input.putScalar(new int[]{2, 1}, 1);
//        labels.putScalar(new int[]{2, 0}, 0);
//        labels.putScalar(new int[]{2, 1}, 1);
//
//        // when both inputs fire, xor is false again - the first output should
//        // fire
//        input.putScalar(new int[]{3, 0}, 0);
//        input.putScalar(new int[]{3, 1}, 1);
//        labels.putScalar(new int[]{3, 0}, 0);
//        labels.putScalar(new int[]{3, 1}, 1);
//        
        // create dataset object
        return new DataSet(input, labels);
	}

}
