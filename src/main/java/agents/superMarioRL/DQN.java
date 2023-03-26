package agents.superMarioRL;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioTimer;
import engine.helper.GameStatus;
import org.datavec.api.records.Buffer;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.cpu.nativecpu.bindings.Nd4jCpu;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.reflect.Array;
import java.net.IDN;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class DQN implements MarioAgent {

    private ReplayBuffer replayBuffer;

    private INDArray prevState;

    private MultiLayerConfiguration networkConfig;
    private MultiLayerNetwork qNetwork;

    private MultiLayerNetwork targetNetwork;

    private int prevAction;

    private double percentCompleted = 0.0;

    private int time = 0;

    private static int seed = 2345870;

    private static final int IMAGE_HEIGHT = 256;

    private static final int IMAGE_WIDTH = 256;

    private static final int CHANNELS = 4;

    private static final int BATCH_SIZE = 128;

    private static final int TRAIN_DELAY = 256; // number of steps between updates to the qNetwork

    private static int trainCounter = 0;

    private static final int UPDATE_DELAY = 1024; // number of steps between the target networks params being updated to those of the q network

    private static int updateCounter = 0;
    private NativeImageLoader loader;

    private String selection;

    public DQN(String selection) {
        this.replayBuffer = new ReplayBuffer();
        this.prevState = null;
        this.networkConfig = createConvModel(seed, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, Agent.ACTION_SPACE);
        this.qNetwork = new MultiLayerNetwork(this.networkConfig);
        this.targetNetwork = new MultiLayerNetwork(this.networkConfig);
        this.qNetwork.init();
        this.targetNetwork.init();
        this.loader = new NativeImageLoader(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS);
        this.selection = selection;
    }
    @Override
    public void initialize(MarioForwardModel model, MarioTimer timer) {

    }

    public MultiLayerConfiguration createConvModel(int seed, int height, int width, int channels, int numActions) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.05))
                .list()
                .layer(0, new ConvolutionLayer.Builder() // 256x256x4 -> 256x256x16
                        .nIn(4)
                        .nOut(8)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // 256x256x16 -> 8x8x16
                        .kernelSize(32,32)
                        .stride(32,32)
                        .build())
                .layer(2, new ConvolutionLayer.Builder() // 8x8x16 -> 8x8x32
                        .nIn(8)
                        .nOut(16)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // 8x8x32 -> 4x4x32
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nIn(256)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new OutputLayer.Builder()
                        .nIn(64)
                        .nOut(numActions)
                        .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
                        .build())

                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        return conf;
    }

    public void trainModel(double gamma) {
        // This should never happen but just in case
        if (this.replayBuffer.size < BATCH_SIZE) {
            return;
        }

        ReplayBuffer samples = this.replayBuffer.sample(BATCH_SIZE);

        ArrayList<INDArray> states = samples.getStateBuffer();
        ArrayList<Integer> actions = samples.getActionBuffer();
        ArrayList<Double> rewards = samples.getRewardBuffer();
        double[][] qOutputs = new double[BATCH_SIZE][Agent.ACTION_SPACE];
        double[] targetOutputs = new double[BATCH_SIZE];
        //** Check if this getDouble() actually works
        ArrayList<INDArray> nextStates = samples.getNextStateBuffer();
        for (int i=0; i<states.size(); i++) {
            qOutputs[i] = this.qNetwork.output(states.get(i).reshape(1, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)).toDoubleVector();
        }
        for (int i=0; i<nextStates.size(); i++) {
            targetOutputs[i] = rewards.get(i) +  gamma*max(this.targetNetwork.output(nextStates.get(i)).get().toDoubleVector());
        }
        for (int i=0; i<actions.size();i++) {
            qOutputs[i][actions.get(i)] = targetOutputs[i];
        }

        INDArray statesArr = Nd4j.create(BATCH_SIZE, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);

        for (int i=0; i<BATCH_SIZE;i++) {
            INDArrayIndex[] indices = {
                    new SpecifiedIndex(i),
                    NDArrayIndex.all(),
            };
            statesArr.put(indices, states.get(i));
        }
        INDArray targetsArr = Nd4j.create(qOutputs);
        System.out.println(Arrays.toString(targetsArr.shape()));

        this.qNetwork.fit(statesArr, targetsArr);
    }

    public double max(double[] values) {
        double maxValue = values[0];
        for (int i=1;i<values.length;i++) {
            if (values[i] > maxValue) {
                maxValue = values[i];
            }
        }
        return maxValue;
    }

    @Override
    public boolean[] getActions(MarioForwardModel model, MarioTimer timer) {
        long startTime = System.currentTimeMillis();
        // While in delay
        //      epsilon greedy selection
        // Save prev_state, prev_action, reward, state to the buffer

        // during updates the prev_state is inputted into the qNetwork and the current (next) state is inputted into the targetNetwork

        // The qNetwork outputs an estimate for the q values of the previous state

        // The targetNetwork outputs an estimate for the q values of the next state

        // Loss is computed: L = MSE(q, r + gamma*q') = sum{(q - (r + gamma*q'))^2}/batch_size

        BufferedImage image = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
        Graphics g = image.getGraphics();
        g.drawImage(model.getImage(), 0, 0, null);
        g.dispose();

        INDArray colourFrame = Nd4j.empty();
        try {
            colourFrame = loader.asMatrix(image);
        } catch (IOException e) {
            System.out.println("Failed to convert BufferedImage to Matrix");
            e.printStackTrace();
        }
        // The matrix is 4 x 256 x 256 -> 4 channels  alpha, r, g, b. Ignore the alpha channel
        long[] shape = colourFrame.shape();
        colourFrame = colourFrame.reshape(shape[1], shape[2], shape[3]);
        INDArray frame = Nd4j.zeros(shape[2], shape[3]);
        INDArrayIndex[] indices = {
                new SpecifiedIndex(1),
              NDArrayIndex.all(),
        };
        frame = colourFrame.get(indices);
        long frameTime = System.currentTimeMillis();
        long grayTime = System.currentTimeMillis();
        boolean[] action;
        this.replayBuffer.addContext(frame);
        int nextActionIndex;
        // If we do not have enough frames for a state and a next state (5 frames in the buffer), return a random action.
        if (this.replayBuffer.getContextBuffer().size() <= CHANNELS) {
            nextActionIndex = randomAction();
            if (this.replayBuffer.getContextBuffer().size() == CHANNELS) {
                this.prevState = this.replayBuffer.getNextStateContext();
            }
            return Agent.ACTION_MAP[nextActionIndex];
        }
        long replayBufferTime = System.currentTimeMillis();
        // This is the (current state) but is the next state in our loss calculation
        INDArray nextState = this.replayBuffer.getNextStateContext();
        nextState = nextState.reshape(1, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);
        long nextStateTime = System.currentTimeMillis();

        // Get predicted action values from the q network
        INDArray nextActionValues = this.qNetwork.output(nextState);

        long feedForwardTime = System.currentTimeMillis();
        nextActionValues = nextActionValues.reshape(Agent.ACTION_SPACE);
        // Perform epsilon greedy selection
        double[] selection = epsilonGreedySelection(nextActionValues.get().toDoubleVector(), Agent.EPSILON);
        nextActionIndex = (int)selection[0];
        double nextActionValue = selection[1];
        action = Agent.ACTION_MAP[nextActionIndex];

        long actionSelectionTime = System.currentTimeMillis();

        long frameT = frameTime - startTime;
        long gray = grayTime - frameTime;
        long replay = replayBufferTime - grayTime;
        long stateTime = nextStateTime - replayBufferTime;
        long feedForward = feedForwardTime - nextStateTime;
        long actionTime = actionSelectionTime - feedForwardTime;

        System.out.println("frame: " + frameT + " Gray: " + gray + " Replay buffer fill: " + replay + " Fetch next state: " + stateTime + " feedforwardtime: " + feedForward + " action selection: " + actionTime);

        // Compute the reward
        double reward = computeReward(model);
        replayBuffer.addEntry(this.prevState, this.prevAction, reward, nextState);

        this.prevState = nextState;
        this.prevAction = nextActionIndex;

        trainCounter++;
        if (trainCounter >= TRAIN_DELAY) {
            trainCounter = 0;
            trainModel(Agent.GAMMA);
        }

        updateCounter++;
        if (updateCounter >= UPDATE_DELAY) {
            updateCounter = 0;
            this.targetNetwork.setParams(this.qNetwork.params());
        }

        return action;
    }

    public int randomAction() {
        Random rand = new Random();
        return rand.nextInt(Agent.ACTION_SPACE);
    }

    public double[] epsilonGreedySelection(double[] actionValuesArray, double epsilon) {
        Random rand = new Random();
        double randomDouble = rand.nextDouble();
        double maxValue = actionValuesArray[0];
        int maxIndex = 0;
        for (int i=1; i<actionValuesArray.length; i++) {
            if (actionValuesArray[i] > maxValue) {
                maxValue = actionValuesArray[i];
                maxIndex = i;
            }
        }
        if (randomDouble <= epsilon) { // With probability of epsilon, select a random action
            maxIndex = rand.nextInt(Agent.ACTION_SPACE);
        }

        return new double[]{(double)maxIndex, maxValue};
    }

//    public INDArray grayscale(INDArray colour) {
//        long[] shape = colour.shape();
//        INDArray gray;
//        int[][] grayPixels = new int[(int)shape[1]][(int)shape[2]];
//        for (int i=0; i<shape[1]; i++) {
//            for (int j=0;j<shape[2];j++) {
//                INDArrayIndex[] indices = {
//                        NDArrayIndex.all(),
//                        new SpecifiedIndex(i),
//                        new SpecifiedIndex(j)
//                };
//                INDArray rgb = colour.get(indices);
//                int[] rgbArray = rgb.get().data().asInt();
//                grayPixels[i][j] = (rgbArray[0] + rgbArray[1] + rgbArray[2])/3;
//            }
//        }
//        gray = Nd4j.create(grayPixels);
//        return gray;
//    }

    @Override
    public String getAgentName() {
        return null;
    }

    public double computeReward(MarioForwardModel model) {
        // Compute Reward
        double reward = 0;
        double progress = model.getCompletionPercentage();
        double difference = progress - this.percentCompleted;
        this.percentCompleted = progress;

        reward += Agent.PROGRESS_REWARD_MULTIPLIER*difference;

//        if (PROGRESS_REWARD_MULTIPLIER* difference < PROGRESS_TOL) {
//            reward -= 1;
//        }

        double deltaT = Agent.TIME_MULTIPLIER*(double)(model.getRemainingTime() - this.time )/100;

        if (this.selection.equals("greedy")) {
            System.out.println("Time: " + this.time);
            System.out.println("Remaining Time: " + model.getRemainingTime());
            System.out.println(deltaT);
        }

        this.time = model.getRemainingTime();
        if (deltaT > 0) { // Safety
            deltaT = 0;
        }

        reward += deltaT;

        if (model.getGameStatus().equals(GameStatus.WIN)) {
            reward += Agent.WIN_REWARD;
            System.out.println("**************************************************");
            System.out.println("VICTORY");
            System.out.println("**************************************************");
        } else if (model.getGameStatus().equals(GameStatus.LOSE)) {
            reward -= Agent.LOSE_PENALTY;
        }
        return reward;
    }

    private static class ReplayBuffer {
        public ArrayList<INDArray> stateBuffer;

        public ArrayList<INDArray> nextStateBuffer;
        public ArrayList<Integer> actionBuffer;
        public ArrayList<Double> rewardBuffer;

        public ArrayList<INDArray> contextBuffer;

        private int size = 0;

        ReplayBuffer() {
            this.stateBuffer = new ArrayList<>();
            this.nextStateBuffer = new ArrayList<>();
            this.actionBuffer = new ArrayList<>();
            this.rewardBuffer = new ArrayList<>();
            this.contextBuffer = new ArrayList<>();
            this.size = 0;
        }

        public ArrayList<INDArray> getStateBuffer() {
            return this.stateBuffer;
        }

        public ArrayList<INDArray> getNextStateBuffer() {
            return this.nextStateBuffer;
        }

        public ArrayList<INDArray> getContextBuffer() {
            return this.contextBuffer;
        }

        public ArrayList<Integer> getActionBuffer() {
            return this.actionBuffer;
        }

        public ArrayList<Double> getRewardBuffer() {
            return this.rewardBuffer;
        }

        public INDArray getPrevState() {
            return this.stateBuffer.get(this.stateBuffer.size()-1);
        }

        public void addEntry(INDArray state, int action, double reward, INDArray nextState) {
            this.stateBuffer.add(state);
            this.actionBuffer.add(action);
            this.rewardBuffer.add(reward);
            this.nextStateBuffer.add(nextState);
            this.size++;
        }

        public void addContext(INDArray frame) {
            this.contextBuffer.add(frame);
        }

        public void addState(INDArray state) {
            this.stateBuffer.add(state);
        }

        public INDArray getNextStateContext() {
            INDArray img = Nd4j.zeros(CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);
            int i = this.contextBuffer.size()-1;
            INDArrayIndex[] indices0 = {
                    new SpecifiedIndex(0),
                    NDArrayIndex.all(),
            };
            img.put(indices0, this.contextBuffer.get(i));
            INDArrayIndex[] indices1 = {
                    new SpecifiedIndex(1),
                    NDArrayIndex.all(),
            };
            img.put(indices1, this.contextBuffer.get(i-1));
            INDArrayIndex[] indices2 = {
                    new SpecifiedIndex(2),
                    NDArrayIndex.all(),
            };
            img.put(indices2, this.contextBuffer.get(i-2));
            INDArrayIndex[] indices3 = {
                    new SpecifiedIndex(3),
                    NDArrayIndex.all(),
            };
            img.put(indices3, this.contextBuffer.get(i-3));
            return img;
        }

        public INDArray getStateContext() {
            INDArray img = Nd4j.zeros(CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);
            int i = this.stateBuffer.size()-1;
            INDArrayIndex[] indices0 = {
                    new SpecifiedIndex(0),
                    NDArrayIndex.all(),
            };
            img.put(indices0, this.stateBuffer.get(i-1));
            INDArrayIndex[] indices1 = {
                    new SpecifiedIndex(1),
                    NDArrayIndex.all(),
            };
            img.put(indices1, this.stateBuffer.get(i-2));
            INDArrayIndex[] indices2 = {
                    new SpecifiedIndex(2),
                    NDArrayIndex.all(),
            };
            img.put(indices2, this.stateBuffer.get(i-3));
            INDArrayIndex[] indices3 = {
                    new SpecifiedIndex(3),
                    NDArrayIndex.all(),
            };
            img.put(indices3, this.stateBuffer.get(i-4));
            return img;
        }

        public ReplayBuffer sample(int batchSize) {
            ReplayBuffer buffer = new ReplayBuffer();
            Random rand = new Random();
            ArrayList<Integer> selected = new ArrayList<>();
            ArrayList<Integer> indices = new ArrayList<>();

            for (int i=0; i<this.size; i++) {
                indices.add(i);
            }

            for (int j=0; j<batchSize; j++) {
                int randomInt = rand.nextInt(indices.size());
                indices.remove(randomInt);
                selected.add(randomInt);
            }

            for (int index : selected) {
                buffer.addEntry(this.stateBuffer.get(index), this.actionBuffer.get(index), this.rewardBuffer.get(index), this.nextStateBuffer.get(index));
            }

            return buffer;
        }
    }
}
