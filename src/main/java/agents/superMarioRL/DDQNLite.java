package agents.superMarioRL;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioTimer;
import engine.helper.GameStatus;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.*;

public class DDQNLite implements MarioAgent {

    private ReplayBuffer replayBuffer;

    private INDArray prevState;

    private MultiLayerConfiguration networkConfig;
    private MultiLayerNetwork qNetwork;

    private MultiLayerNetwork targetNetwork;

    private int prevAction;

    private double percentCompleted = 0.0;

    private int time = 0;

    private int elevation = 0;

    public static boolean onTile = true;

    private static int seed = 2345870;

    private static final int IMAGE_HEIGHT = 16;

    private static final int IMAGE_WIDTH = 16;

    private static final int CHANNELS = 4;

    private static final int BATCH_SIZE = 32;

    private static final int TRAIN_DELAY = 128; // number of steps between updates to the qNetwork

    private static int trainCounter = 0;

    private static final int UPDATE_DELAY = 256; // number of steps between the target networks params being updated to those of the q network

    private static int updateCounter = 0;

    private static final int BUFFER_CAPACITY = 2048;
    private NativeImageLoader loader;

    private String selection;

    private int elimByStomp = 0;

    private int elimByFire = 0;

    public double bestProgress = 0;

    public ArrayList<String[]> performanceData;

    public int gapSize = 0;

    public static final int GAP_REWARD = 1;

    public DDQNLite(String selection) {
        this.replayBuffer = new ReplayBuffer();
        this.prevState = null;
//        this.networkConfig = createConvModel(seed, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, Agent.ACTION_SPACE);
        this.networkConfig = createNNModel(seed, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, Agent.ACTION_SPACE);
        this.qNetwork = new MultiLayerNetwork(this.networkConfig);
        this.targetNetwork = new MultiLayerNetwork(this.networkConfig);
        this.qNetwork.init();
        this.targetNetwork.init();
        this.loader = new NativeImageLoader(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS);
        this.selection = selection;
        this.performanceData = new ArrayList<>();
    }
    @Override
    public void initialize(MarioForwardModel model, MarioTimer timer) {
        resetProgress(model);
    }

    public void resetPerformanceData() {
        this.performanceData = new ArrayList<>();
    }
    public void resetProgress(MarioForwardModel model) {
        this.percentCompleted = model.getCompletionPercentage();
        this.elimByStomp = 0;
        this.elimByFire = 0;
        this.time = model.getRemainingTime();
        this.bestProgress = percentCompleted;
        this.elevation = model.getMarioScreenTilePos()[1];
        this.gapSize = 0;
    }

    public ArrayList<String[]> getPerformanceData() {
        return this.performanceData;
    }

    public void updateOnTile(int[] marioPos, int[][] tiles) {
        if (marioPos[1] < Agent.CANVAS_MAX && tiles[marioPos[0]][marioPos[1]+1] > 0) {
            onTile = true;
        } else {
            onTile = false;
        }
    }

    public boolean updateOverGap(int[] marioPos, int[][] tiles) {
        for (int i=marioPos[1]+1; i<Agent.CANVAS_MAX; i++) {
            if (tiles[marioPos[0]][i] > 0) {
                return false;
            }
        }
        return true;
    }

    public MultiLayerConfiguration createConvModel(int seed, int height, int width, int channels, int numActions) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.05))
                .list()
                .layer(0, new ConvolutionLayer.Builder() // 16x16x4 -> 16x16x8
                        .nIn(4)
                        .nOut(8)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // 16x16x8 -> 4x4x8
                        .kernelSize(4,4)
                        .stride(4,4)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new OutputLayer.Builder()
                        .nIn(64)
                        .nOut(numActions)
                        .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
                        .build())
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        return conf;
    }

    public MultiLayerConfiguration createNNModel(int seed, int height, int width, int channels, int numActions) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.1))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(1024)
                        .nOut(768)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(768)
                        .nOut(576)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(576)
                        .nOut(432)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(432)
                        .nOut(324)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nIn(324)
                        .nOut(243)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new DenseLayer.Builder()
                        .nIn(243)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(6, new OutputLayer.Builder()
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

        LinkedList<INDArray> states = samples.getStateBuffer();
        LinkedList<Integer> actions = samples.getActionBuffer();
        LinkedList<Double> rewards = samples.getRewardBuffer();
        double[][] qOutputs = new double[BATCH_SIZE][Agent.ACTION_SPACE];
        int[] argmax = new int[BATCH_SIZE];
        double[] targetOutputs = new double[BATCH_SIZE];
        //** Check if this getDouble() actually works
        LinkedList<INDArray> nextStates = samples.getNextStateBuffer();
        int i = 0;
        for (INDArray state : states) {
            qOutputs[i] = this.qNetwork.output(state.reshape(1, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)).toDoubleVector();
            i++;
        }
        i = 0;
        for (INDArray nextState : nextStates) {
            double[] output = this.qNetwork.output(nextState).toDoubleVector();
            int maxIndex = 0;
            double maxValue = output[0];
            for (int j=1;j<output.length;j++) {
                if (output[j] > maxValue) {
                    maxValue = output[j];
                    maxIndex = j;
                }
            }
            argmax[i] = maxIndex;
            i++;
        }
        i = 0;
        Iterator<Double> rewardsIter = rewards.iterator();
        Iterator<INDArray> nextStatesIter = nextStates.iterator();
        while (rewardsIter.hasNext()) {
            targetOutputs[i] = rewardsIter.next() +  gamma*this.targetNetwork.output(nextStatesIter.next()).get().toDoubleVector()[argmax[i]];
            i++;
        }
        i = 0;
        for (Integer action : actions) {
            qOutputs[i][action] = targetOutputs[i];
            i++;
        }

        INDArray statesArr = Nd4j.create(BATCH_SIZE, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);

        for (i=0; i<BATCH_SIZE;i++) {
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


        // The matrix is 4 x 256 x 256 -> 4 channels  alpha, r, g, b. Ignore the alpha channel
        if (Agent.selection.equals("epsilongreedytest")) {
            int[][] scene = model.getMarioCompleteObservation();
            for (int i=0;i<scene.length;i++) {
                System.out.println(Arrays.toString(scene[i]));
            }
        }
        INDArray frame = Nd4j.create(model.getMarioCompleteObservation());
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
        double[] selection = epsilonGreedySelection(nextActionValues.get().toDoubleVector(), Agent.epsilon);
        nextActionIndex = (int)selection[0];
        double nextActionValue = selection[1];
        action = Agent.ACTION_MAP[nextActionIndex];

        long actionSelectionTime = System.currentTimeMillis();

        // Update whether mario is on the ground
        updateOnTile(model.getMarioScreenTilePos(), model.getScreenSceneObservation());
         if (updateOverGap(model.getMarioScreenTilePos(), model.getScreenSceneObservation())) {
            this.gapSize++;
         }
        // Compute the reward
        double reward = computeReward(model);
        System.out.println("Reward: " + reward);
        replayBuffer.addEntry(this.prevState, this.prevAction, reward, nextState);

        this.prevState = nextState;
        this.prevAction = nextActionIndex;
        long trainStartTime = System.currentTimeMillis();
        trainCounter++;
        if (trainCounter >= TRAIN_DELAY) {
            trainCounter = 0;
            trainModel(Agent.GAMMA);
        }
        long trainEndTime = System.currentTimeMillis();

        updateCounter++;
        if (updateCounter >= UPDATE_DELAY) {
            updateCounter = 0;
            this.targetNetwork.setParams(this.qNetwork.params());
        }
        long updateTime = System.currentTimeMillis();

        long stateTime = nextStateTime - replayBufferTime;
        long feedForward = feedForwardTime - nextStateTime;
        long trainTime = trainEndTime - trainStartTime;
        long updateT = updateTime - trainEndTime;

//        System.out.println("Gray: " + gray + " Replay buffer fill: " + replay + " Fetch next state: " + stateTime + " feedforwardtime: " + feedForward + " traintime: " + trainTime + " updatetime: " + updateT);

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

        double elevationReward = 0;
        double gapReward = 0;
//        if (this.selection.equals("epsilongreedytest")) {
//            System.out.println("GROUND: "+ model.isMarioOnGround() + " " + marioPos[1]);
//        }
        if (this.onTile && progress > this.bestProgress) {
            if (model.getMarioScreenTilePos()[1] < this.elevation) {
                elevationReward += Math.pow(this.elevation - model.getMarioScreenTilePos()[1], 2);
                if (this.selection.equals("epsilongreedytest")) {
                    System.out.println("ELEVATION REWARD: " + elevationReward + " " + model.getMarioScreenTilePos()[1] + " " + this.elevation);
                }
            }
            this.elevation = model.getMarioScreenTilePos()[1];
            if (this.gapSize > 0) {
                gapReward += this.gapSize;
                this.gapSize = 0;
            }
        }


        reward += Agent.ELEVATED_REWARD*elevationReward;
        reward += GAP_REWARD*gapReward;

        int stompElim = 0;
        if (model.getKillsByStomp() > this.elimByStomp) {
            stompElim = model.getKillsByStomp() - this.elimByStomp;
            this.elimByStomp = model.getKillsByStomp();
        }
        reward += Agent.ELIM_REWARD*stompElim;

        int fireElim = 0;
        if (model.getKillsByFire() > this.elimByFire) {
            fireElim = model.getKillsByFire() - this.elimByFire;
            this.elimByFire = model.getKillsByFire();
        }
        reward += Agent.ELIM_REWARD*fireElim;

        if (model.getGameStatus().equals(GameStatus.WIN)) {
            reward += Agent.WIN_REWARD;
            System.out.println("**************************************************");
            System.out.println("VICTORY");
            System.out.println("**************************************************");
        } else if (model.getGameStatus().equals(GameStatus.LOSE)) {
            reward -= Agent.LOSE_PENALTY;
        }

        if (progress > this.bestProgress) {
            this.bestProgress = progress;
        }

        if (!model.getGameStatus().equals(GameStatus.RUNNING)) {
            String[] data = new String[3];
            data[0] = String.valueOf(model.getCompletionPercentage());
            data[1] = model.getGameStatus().toString();
            data[2] = String.valueOf(((double)(400000 - model.getRemainingTime())/1000));
            performanceData.add(data);
        }

        return reward;
    }

    private static class ReplayBuffer {
        public LinkedList<INDArray> stateBuffer;

        public LinkedList<INDArray> nextStateBuffer;
        public LinkedList<Integer> actionBuffer;
        public LinkedList<Double> rewardBuffer;

        public LinkedList<INDArray> contextBuffer;

        private int size;
        private int capacity;
        private int contextSize;

        ReplayBuffer() {
            this.stateBuffer = new LinkedList<>();
            this.nextStateBuffer = new LinkedList<>();
            this.actionBuffer = new LinkedList<>();
            this.rewardBuffer = new LinkedList<>();
            this.contextBuffer = new LinkedList<>();
            this.size = 0;
            this.contextSize = 0;
            this.capacity = BUFFER_CAPACITY;
        }

        public LinkedList<INDArray> getStateBuffer() {
            return this.stateBuffer;
        }

        public LinkedList<INDArray> getNextStateBuffer() {
            return this.nextStateBuffer;
        }

        public LinkedList<INDArray> getContextBuffer() {
            return this.contextBuffer;
        }

        public LinkedList<Integer> getActionBuffer() {
            return this.actionBuffer;
        }

        public LinkedList<Double> getRewardBuffer() {
            return this.rewardBuffer;
        }

        public void addEntry(INDArray state, int action, double reward, INDArray nextState) {
            if (this.size >= this.capacity) {
                this.stateBuffer.removeLast();
                this.actionBuffer.removeLast();
                this.rewardBuffer.removeLast();
                this.nextStateBuffer.removeLast();
            } else {
                this.size++;
            }
            this.stateBuffer.addFirst(state);
            this.actionBuffer.addFirst(action);
            this.rewardBuffer.addFirst(reward);
            this.nextStateBuffer.addFirst(nextState);
        }

        public void addContext(INDArray frame) {
            if (this.contextSize >= this.capacity) {
                this.contextBuffer.removeLast();
            } else {
                contextSize++;
            }
            this.contextBuffer.addFirst(frame);
        }

        public void addState(INDArray state) {
            this.stateBuffer.add(state);
        }

        public INDArray getNextStateContext() {
            INDArray img = Nd4j.zeros(CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);
            INDArrayIndex[] indices0 = {
                    new SpecifiedIndex(0),
                    NDArrayIndex.all(),
            };
            img.put(indices0, this.contextBuffer.get(0));
            INDArrayIndex[] indices1 = {
                    new SpecifiedIndex(1),
                    NDArrayIndex.all(),
            };
            img.put(indices1, this.contextBuffer.get(1));
            INDArrayIndex[] indices2 = {
                    new SpecifiedIndex(2),
                    NDArrayIndex.all(),
            };
            img.put(indices2, this.contextBuffer.get(2));
            INDArrayIndex[] indices3 = {
                    new SpecifiedIndex(3),
                    NDArrayIndex.all(),
            };
            img.put(indices3, this.contextBuffer.get(3));
            return img;
        }

        public INDArray getStateContext() {
            INDArray img = Nd4j.zeros(CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);
            INDArrayIndex[] indices0 = {
                    new SpecifiedIndex(0),
                    NDArrayIndex.all(),
            };
            img.put(indices0, this.stateBuffer.get(1));
            INDArrayIndex[] indices1 = {
                    new SpecifiedIndex(1),
                    NDArrayIndex.all(),
            };
            img.put(indices1, this.stateBuffer.get(2));
            INDArrayIndex[] indices2 = {
                    new SpecifiedIndex(2),
                    NDArrayIndex.all(),
            };
            img.put(indices2, this.stateBuffer.get(3));
            INDArrayIndex[] indices3 = {
                    new SpecifiedIndex(3),
                    NDArrayIndex.all(),
            };
            img.put(indices3, this.stateBuffer.get(4));
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
