package agents.superMarioRL;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioTimer;
import engine.helper.GameStatus;
import engine.sprites.FlowerEnemy;
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

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.*;

public class PPO implements MarioAgent {

    private RolloutBuffer rolloutBuffer;

    private INDArray prevState;

    private MultiLayerConfiguration networkConfig;
    private MultiLayerNetwork actor;

    private MultiLayerNetwork critic;

    private int prevAction;

    private double percentCompleted = 0.0;

    private int time = 0;

    private int elevation = 0;

    public static boolean onTile = true;

    private static int seed = 2345870;

    private static final int SOURCE_IMAGE_HEIGHT = 256;

    private static final int SOURCE_IMAGE_WIDTH = 256;

    private static final int IMAGE_HEIGHT = 84;

    private static final int IMAGE_WIDTH = 84;

    private static final int CHANNELS = 4;

    private static final int BATCH_SIZE = 32;

    private static final int TRAIN_DELAY = 1; // number of trajectories

    private static int trainCounter = 0;
    private static final int UPDATE_DELAY = 1; // number of trajectories
    private static final double CLIP = 0.2;
    private static int updateCounter = 0;

    private static final int BUFFER_CAPACITY = 1024;
    private NativeImageLoader loader;

    private String selection;

    private int elimByStomp = 0;

    private int elimByFire = 0;
    public double bestProgress = 0;
    public ArrayList<String[]> performanceData;

    public int gapSize = 0;

    public static final int GAP_REWARD = 1;

    public static final double GAMMA = 0.9;

    public PPO(String selection) {
        this.rolloutBuffer = new RolloutBuffer();
        this.prevState = null;
        MultiLayerConfiguration actorConfig = createActorModel(seed, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, Agent.ACTION_SPACE);
        MultiLayerConfiguration criticConfig = createCriticModel(seed, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS);
        this.actor = new MultiLayerNetwork(actorConfig);
        this.critic = new MultiLayerNetwork(criticConfig);
        this.actor.init();
        this.critic.init();
        this.loader = new NativeImageLoader(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS);
        this.selection = selection;
        this.performanceData = new ArrayList<>();
    }
    @Override
    public void initialize(MarioForwardModel model, MarioTimer timer) {
        resetProgress(model);
    }

    public void resetProgress(MarioForwardModel model) {
        this.percentCompleted = model.getCompletionPercentage();
        this.elimByStomp = 0;
        this.elimByFire = 0;
        this.bestProgress = percentCompleted;
        this.time = model.getRemainingTime();
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

    public void resetPerformanceData() {
        this.performanceData = new ArrayList<>();
    }

    public MultiLayerConfiguration createActorModel(int seed, int height, int width, int channels, int numActions) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.1))
                .list()
                .layer(0, new ConvolutionLayer.Builder() // 84x84x4 -> 80x80x16
                        .nIn(4)
                        .nOut(8)
                        .kernelSize(5, 5)
                        .stride(1, 1)
                        .padding(0, 0)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // 80x80x16 -> 8x8x16
                        .kernelSize(10,10)
                        .stride(10,10)
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
                .layer(4, new ConvolutionLayer.Builder() // 4x4x32 -> 4x4x32
                        .nIn(16)
                        .nOut(16)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nIn(256)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new OutputLayer.Builder()
                        .nIn(128)
                        .nOut(numActions)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
                        .build())

                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        return conf;
    }

    public MultiLayerConfiguration createCriticModel(int seed, int height, int width, int channels) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.1))
                .list()
                .layer(0, new ConvolutionLayer.Builder() // 84x84x4 -> 80x80x16
                        .nIn(4)
                        .nOut(8)
                        .kernelSize(5, 5)
                        .stride(1, 1)
                        .padding(0, 0)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // 80x80x16 -> 8x8x16
                        .kernelSize(10,10)
                        .stride(10,10)
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
                .layer(4, new ConvolutionLayer.Builder() // 4x4x32 -> 4x4x32
                        .nIn(16)
                        .nOut(16)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nIn(256)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new OutputLayer.Builder()
                        .nIn(128)
                        .nOut(1)
                        .activation(Activation.IDENTITY)
                        .lossFunction(LossFunctions.LossFunction.SQUARED_LOSS)
                        .build())
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();
        return conf;
    }

    public void trainActor(double gamma) {
        // This should never happen but just in case
        if (this.rolloutBuffer.size < BATCH_SIZE) {
            return;
        }

        RolloutBuffer samples = this.rolloutBuffer.sample(BATCH_SIZE);




        LinkedList<INDArray> states = samples.getStateBuffer();
        LinkedList<Integer> actions = samples.getActionBuffer();
        LinkedList<Double> rewards = samples.getRewardBuffer();
        LinkedList<Double> advantages = samples.getAdvantageBuffer();
        LinkedList<Double> oldLogProbabilities = samples.getLogProbabilityBuffer();
        ArrayList<Double> policyRatios = new ArrayList<>();

        // First obtain the new log probabilities
        Iterator<INDArray> statesIter = states.iterator();
        Iterator<Integer> actionsIter = actions.iterator();
        Iterator<Double> oldLogIter = oldLogProbabilities.iterator();
        while (statesIter.hasNext()) {
            double[] actionDistribution = this.actor.output(statesIter.next()).toDoubleVector();
            policyRatios.add(Math.exp(Math.log(actionDistribution[actionsIter.next()] - oldLogIter.next())));
        }



        double[] loss = new double[BATCH_SIZE];
        int i = 0;
        for (double a : advantages) {
            double g = a >= 0 ? (1 + CLIP)*a : (1 - CLIP)*a;
            loss[i] = Math.min(policyRatios.get(i)*a, g);
            i++;
        }

        double[][] qOutputs = new double[BATCH_SIZE][Agent.ACTION_SPACE];
        //** Check if this getDouble() actually works
        i = 0;
        for (INDArray state : states) {
            qOutputs[i] = this.actor.output(state.reshape(1, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)).toDoubleVector();
            i++;
        }
        i = 0;
        for (Integer action : actions) {
            qOutputs[i][action] += loss[i];
            i++;
        }

        INDArray statesArr = Nd4j.create(BATCH_SIZE, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);
        statesIter = states.iterator();
        for (i=0; i<BATCH_SIZE;i++) {
            INDArrayIndex[] indices = {
                    new SpecifiedIndex(i),
                    NDArrayIndex.all(),
            };
            statesArr.put(indices, statesIter.next());
        }
        INDArray targetsArr = Nd4j.create(qOutputs);
        System.out.println(Arrays.toString(targetsArr.shape()));

        this.actor.fit(statesArr, targetsArr);
    }

    public void trainCritic() {

        LinkedList<INDArray> states = this.rolloutBuffer.getStateBuffer();
        LinkedList<Double> returns = this.rolloutBuffer.getReturnBuffer();

        INDArray statesArr = Nd4j.create(states.size(), CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);
        Iterator<INDArray> statesIter = states.iterator();
        for (int i=0; i<BATCH_SIZE;i++) {
            INDArrayIndex[] indices = {
                    new SpecifiedIndex(i),
                    NDArrayIndex.all(),
            };
            statesArr.put(indices, statesIter.next());
        }

        System.out.println(states.size());
        System.out.println(returns.size());

        INDArray returnsArr = Nd4j.create(returns);

        returnsArr.reshape(returns.size());
        statesArr.reshape(states.size(), CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);

        this.critic.fit(statesArr, returnsArr);

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

        // Convert to grayscale and re-size: 256x256 -> 84x84
        long frameTime = System.currentTimeMillis();
        Graphics g = image.getGraphics();
        g.drawImage(model.getImage(), 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, null);
        g.dispose();

//        new JFrame("gray") {
//            {
//                final JLabel label = new JLabel("", new ImageIcon(image), 0);
//                add(label);
//                pack();
//                setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
//                setVisible(true);
//            }
//        };



        long grayTime = System.currentTimeMillis();
        INDArray frame = Nd4j.empty();
        try {
            frame = loader.asMatrix(image);
        } catch (IOException e) {
            System.out.println("Failed to convert BufferedImage to Matrix");
            e.printStackTrace();
        }
        // The matrix is 4 x 256 x 256 -> 4 channels  alpha, r, g, b. Ignore the alpha channel
        long[] shape = frame.shape();
        frame = frame.reshape(shape[1], shape[2], shape[3]);
        INDArrayIndex[] indices = {
                new SpecifiedIndex(1),
              NDArrayIndex.all(),
        };
        frame = frame.get(indices);

        boolean[] action;

        this.rolloutBuffer.addContext(frame);

        int actionIndex;
        // If we do not have enough frames for a state and a next state (5 frames in the buffer), return a random action.
        if (this.rolloutBuffer.getContextBuffer().size() <= CHANNELS) {
            actionIndex = randomAction();
            return Agent.ACTION_MAP[actionIndex];
        }

        long replayBufferTime = System.currentTimeMillis();
        // This is the (current state) but is the next state in our loss calculation

        INDArray state = this.rolloutBuffer.getStateContext();
        state = state.reshape(1, CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH);
        long nextStateTime = System.currentTimeMillis();

        // Get predicted action values from the q network
        INDArray actorOutput = this.actor.output(state);
        INDArray criticOutput = this.critic.output(state);

        long feedForwardTime = System.currentTimeMillis();
        actorOutput = actorOutput.reshape(Agent.ACTION_SPACE);

        double[] actionDistribution = actorOutput.get().toDoubleVector();
        double value = criticOutput.toDoubleVector()[0];
        // Perform epsilon greedy selection
        System.out.println(Arrays.toString(actionDistribution));
        actionIndex = actionSelection(actionDistribution);
        action = Agent.ACTION_MAP[actionIndex];
        double logProbability = Math.log(actionDistribution[actionIndex]);

        // Update whether mario is on the ground
        updateOnTile(model.getMarioScreenTilePos(), model.getScreenSceneObservation());
        if (updateOverGap(model.getMarioScreenTilePos(), model.getScreenSceneObservation())) {
            this.gapSize++;
        }

        // Compute the reward
        double reward = computeReward(model);
        System.out.println("Reward: " + reward);
        rolloutBuffer.addEntry(state, actionIndex, reward, value, logProbability);
        if (!model.getGameStatus().equals(GameStatus.RUNNING)) {
            rolloutBuffer.completeTrajectory(GAMMA);
            updateCounter++;
            trainCounter++;
        }

        long trainStartTime = System.currentTimeMillis();
        if (trainCounter >= TRAIN_DELAY) {
            trainCounter = 0;
            trainActor(GAMMA);
        }
        long trainEndTime = System.currentTimeMillis();


        if (updateCounter >= UPDATE_DELAY) {
            updateCounter = 0;
            trainCritic();
        }
        long updateTime = System.currentTimeMillis();

        long gray = grayTime - frameTime;
        long replay = replayBufferTime - grayTime;
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

    public int actionSelection(double[] actionDistribution) {
        double thresh = Math.random();
        double cumulativeProbability = 0;
        for (int i=0;i<actionDistribution.length;i++) {
            cumulativeProbability += actionDistribution[i];
            if (cumulativeProbability >= thresh) {
                return i;
            }
        }
        return actionDistribution.length-1;
    }

    public double entropy(double[] actionDistribution) {
        double h = 0;
        for (double p : actionDistribution) {
            h -= p*Math.log(p);
        }
        return h;
    }

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

    private static class RolloutBuffer {
        public LinkedList<INDArray> stateBuffer;
        public LinkedList<Integer> actionBuffer;
        public LinkedList<Double> rewardBuffer;
        public LinkedList<INDArray> contextBuffer;
        public LinkedList<Double> advantageBuffer;
        public LinkedList<Double> logProbabilityBuffer;
        public LinkedList<Double> valueBuffer;

        public LinkedList<Double> returnBuffer;
        private int size;
        private int trajectoryPointer;
        private int capacity;
        private int contextSize;

        RolloutBuffer() {
            this.stateBuffer = new LinkedList<>();
            this.actionBuffer = new LinkedList<>();
            this.rewardBuffer = new LinkedList<>();
            this.contextBuffer = new LinkedList<>();
            this.advantageBuffer = new LinkedList<>();
            this.logProbabilityBuffer = new LinkedList<>();
            this.valueBuffer = new LinkedList<>();
            this.returnBuffer = new LinkedList<>();
            this.size = 0;
            this.contextSize = 0;
            this.trajectoryPointer = -1;
            this.capacity = BUFFER_CAPACITY;
        }

        public LinkedList<INDArray> getStateBuffer() {
            return this.stateBuffer;
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

        public LinkedList<Double> getAdvantageBuffer() { return this.advantageBuffer; }

        public LinkedList<Double> getLogProbabilityBuffer() { return this.logProbabilityBuffer; }

        public LinkedList<Double> getReturnBuffer() { return this.returnBuffer; }

        public void addEntry(INDArray state, int action, double reward, double value, double logProbability) {
            if (this.size >= this.capacity) {
                this.stateBuffer.removeLast();
                this.actionBuffer.removeLast();
                this.rewardBuffer.removeLast();
                this.valueBuffer.removeLast();
                this.logProbabilityBuffer.removeLast();
            } else {
                this.size++;
            }
            this.stateBuffer.addFirst(state);
            this.actionBuffer.addFirst(action);
            this.rewardBuffer.addFirst(reward);
            this.valueBuffer.addFirst(value);
            this.logProbabilityBuffer.addFirst(logProbability);
            this.trajectoryPointer++;
            if (this.trajectoryPointer >= this.size) {
                this.trajectoryPointer = this.size-1;
            }
        }

        public void addContext(INDArray frame) {
            if (this.contextSize >= this.capacity) {
                this.contextBuffer.removeLast();
            } else {
                contextSize++;
            }
            this.contextBuffer.addFirst(frame);
        }

        public void completeTrajectory(double gamma) {
            double discountedReturn = 0;
            for (int i=this.trajectoryPointer; i>0;i--) {
                this.advantageBuffer.addFirst(rewardBuffer.get(i) + gamma*this.valueBuffer.get(i-1) - this.valueBuffer.get(i));
            }
            double[] returns = new double[trajectoryPointer+1];
            for(int i=0;i<=trajectoryPointer;i++) {
                discountedReturn += rewardBuffer.get(i);
                returns[i] = discountedReturn;
                discountedReturn *= gamma;
            }

            for (int i=trajectoryPointer; i>=0;i--) {
                this.returnBuffer.addFirst(returns[i]);
            }
            this.trajectoryPointer = 0;


        }

        public void addState(INDArray state) {
            this.stateBuffer.add(state);
        }

        public INDArray getStateContext() {
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

        public RolloutBuffer sample(int batchSize) {
            RolloutBuffer buffer = new RolloutBuffer();
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
                buffer.addEntry(this.stateBuffer.get(index), this.actionBuffer.get(index), this.rewardBuffer.get(index), this.valueBuffer.get(index), this.logProbabilityBuffer.get(index));
            }

            return buffer;
        }
    }
}
