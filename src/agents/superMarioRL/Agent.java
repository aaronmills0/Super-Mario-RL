package agents.superMarioRL;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioTimer;
import engine.helper.GameStatus;
import engine.helper.MarioActions;
import engine.sprites.Mario;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.lang.Math;

/**
 * State Space:
 * The game is represented by a grid of 22 x 22 = 484 tiles
 *
 * Each tile can be empty (background), contain a tile (solid), or contain some kind of object of which there are 16 (plus 1 undef?)
 *
 * Out of these objects we have enemies (11):
 * Goomba, Winged Goomba, Red Koopa, Winged Red Koopa, Green Koopa, Winged Green Koopa, Spiky, Winged Spiky, Bullet Bill, Fireball, Enemy Flower
 * items (4):
 * Mushroom, Fire Flower, Shell, Life Mushroom
 * and Mario (1)
 *
 *In addition, there are many different types of tiles (9 (10 if Block All is included)):
 * Block Upper, Block Lower, Block All, Special, Life, Bumpable, Breakable, Pickable, Animated, Spawner
 *
 *
 * Other state considerations:
 *
 *      1. Mario has 3 different states: small:0, large:1, and fire:2
 *         model.getMarioMode() -> int
 *      2. If mario can jump higher (can mario jump higher by holding down the jump button)
 *         model.getMarioCanJumpHigher() -> boolean
 *      3. If mario is on the ground
 *         model.isMarioOnGround() -> boolean
 *      4. If mario is able to jump
 *         model.mayMarioJump() -> boolean
 *
 * Some function that may be useful in computing the state
 *
 *      1. Get Mario's position (x, y) in terms of tiles
 *         model.getMarioScreenTilePos() -> int[]
 *      2. The current objects (not enemies) on the screen as a 2D tile grid around mario with a detail value of 1
 *         2D grid where each tile contain either 0 which means it is empty or a value that reflect the type of the
 *         tile in that area.
 *         model.getMarioSceneObservation() -> int[][]
 *      3. The current enemies on the screen as a 2D tile grid around mario with a detail value of 0
 *         2D grid where each tile contain either 0 to indicate no enemy or a number to indicate a certain enemy.
 *         model.
 *******************************************************************************************
 * The state will first be encoded using the encoding by Liao, Yi, and Yang
 *      1. Mario Mode: 0 (small), 1 (big), 2 (fire) [2 bits]
 *      2. Direction: The direction of Mario's velocity 9 possible values -
 *         8 for different directions and one to indicate no movement [4 bits]
 *      3. Stuck: A 0 or a 1 indicating if Mario has not moved for several frames (i.e., no movement is indicated by the direction attribute) [1 bit]
 *      4. On Ground: A 0 or a 1 [1 bit]
 *      5. Can Jump: A 0 or a 1 [1 bit]
 *      6. Collided With Enemy: A 0 or a 1 [1 bit]
 *      7. Enemies Nearby: Indicates whether there is an enemy in the 8 directions within 8 (small) or 10 (large/fire) tiles surrounding Mario.
 *         There is a 3x3 box for small Mario and a 4x3 box for large/fire Mario. [8 bits]
 *      8. Enemies Midrange: Indicates whether there is an enemy in the 8 directions in the midrange around Mario (excludes close enemies).
 *         There is a 7x7 box for small Mario and a 8x7 box for large/fire Mario. [8 bits]
 *      9. Enemies Far: Indicates whether there is an enemy in the 8 directions far from Mario (excludes midrange and close enemies).
 *         There is a 11x11 box for small Mario and a 12x11 box for large/fire Mario. [8 bits]
 *     10. Killed Enemy With Stomp: 0 or 1 [1 bit]
 *     11. Killed Enemy With Fire: 0 or 1 [1 bit]
 *     12. Obstacles: Indicates whether there exists obstacles in the 4 tiles in front of Mario. [4 bits]
 *
 *     With the above encoding there are 2^40 possible states (paper get 2^39 - unsure how), therefore
 *     a HashTable is used such that only states that are actually observed are stored.
 *
 *******************************************************************************************
 *
 * Action Space
 *
 * The agent returns an array of booleans of length of 5:
 * [LEFT, RIGHT, DOWN, SPEED, JUMP]
 * In reality, there many combination of these actions:
 * {LEFT, RIGHT, DOWN, NONE} x {JUMP, NOJUMP} x {SPEED, NOSPEED} (SPEED is only available with fire Mario)
 *
 * This means there are 16 actions combinations in total, encoded as a 5 bit boolean (The paper excluded DOWN leaving only 12 action combinations)
 *
 * We can reduce this to 14 by considering that is does not make sense to perform DOWN and JUMP together
 * (If necessary, maybe we can further reduce this as it may not make sense to perform NONE and SPEED together) **
 *
 * (RIGHT, NOJUMP, SPEED) : [false, true, false, true, false]
 * (RIGHT, NOJUMP, NOSPEED) : [false, true, false, false, false]
 * (RIGHT, JUMP, SPEED) : [false, true, false, true, true]
 * (RIGHT, JUMP, NOSPEED) : [false, true, false, false, true]
 * (LEFT, NOJUMP, SPEED) : [true, false, false, true, false]
 * (LEFT, NOJUMP, NOSPEED) : [true, false, false, false, false]
 * (LEFT, JUMP, SPEED) : [true, false, false, true, true]
 * (LEFT, JUMP, NOSPEED) : [true, false, false, false, true]
 * (DOWN, NOJUMP, SPEED) : [false, false, true, true, false]
 * (DOWN, NOJUMP, NOSPEED) : [false, false, true, false, false]
 * (NONE, NOJUMP, SPEED) : [false, false, false, true, false] **
 * (NONE, NOJUMP, NOSPEED) : [false, false, false, false, false]
 * (NONE, JUMP, SPEED) : [false, false, false, true, true] **
 * (NONE, JUMP, NOSPEED) : [false, false, false, false, true]
 *
 * Reward:
 *
 * Some functions that we may be able to use to calculate reward
 *      1. Game Status (WIN, LOSE, TIMEOUT, RUNNING)
 *         model.getGameStatus() -> Enum(GameStatus)
 *      2. Completion Percentage
 *         model.getCompletionPercentage() -> float (between 0 and 1)
 *      3. Total number of kills by fireballs
 *         model.getKillsByFire() -> int
 *      4. Total number of kills by stomping
 *         model.getKillsByStomp() -> int
 *      5. Total number of kills by shell
 *         model.getKillsByShell() -> int
 *      6. Total number of mushrooms collected
 *         model.getNumCollectedMushrooms() -> int
 *      7. Total number of fire flowers collected
 *         model.getNumCollectedFireFlower() -> int
 *      8. Total number of coins collected
 *         model.getNumCollectedCoins() -> int
 *
 * The basic idea is to reward progress in the level.
 *
 * The most important aspect to reward is completion of the level.
 *
 * My first thoughts are to reward winning heavily and to reward increasing the completion percentage by a small amount
 *
 * Moving backward and collision with enemies and getting stuck will be negatively rewarded
 *
 */


public class Agent implements MarioAgent {

    private boolean[] actions; // boolean with a true false value. true if the action is to be taken and false if not
    public HashMap<String, double[]> Q; // action-value estimate

    public HashMap<String, double[]> Q2; // action-value estimate used for Double Q-Learning

    private String algorithm = "qlearning";

    private String selection = "epsilongreedy";

    // Keep track of certain attributes to assist in computing reward and or state
    private double percentCompleted = 0.0;
    public int stuckCounter = 0;
    private final int STUCK_LIMIT = 48; // stuck for 48 frames
    private int prevMarioMode = 0;
    public int elimByStomp = 0;
    public int elimByFire = 0;
    private final int NEARBY_RADIUS = 3;
    private final int MIDRANGE_RADIUS = 7;
    private final int FARRANGE_RADIUS = 11;
    private final int PROGRESS_REWARD_MULTIPLIER = 1000;
    private final int ELEVATED_REWARD = 1;
    private final int WIN_REWARD = 500;

    private final int LOSE_PENALTY = 25;
    private final int ELIM_REWARD = 1;
    private final int COLLISION_PENALTY = 5;
    private final double EPSILON = 0.3;
    private final double GAMMA = 0.6;
    private final double ALPHA = 0.15;
    private final int ACTION_SPACE = 14;
    private int time = 0;

    private int elevation = 0;

    private final double TIME_MULTIPLIER = 0.5;
    public static boolean onTile = true;

    public double bestProgress = 0;

    public ArrayList<ArrayList<String[]>> performanceData;

    public ArrayList<String[]> episodeData;

    /*

     * (RIGHT, NOJUMP, SPEED) : [false, true, false, true, false]
     * (RIGHT, NOJUMP, NOSPEED) : [false, true, false, false, false]
     * (RIGHT, JUMP, SPEED) : [false, true, false, true, true]
     * (RIGHT, JUMP, NOSPEED) : [false, true, false, false, true]
     * (LEFT, NOJUMP, SPEED) : [true, false, false, true, false]
     * (LEFT, NOJUMP, NOSPEED) : [true, false, false, false, false]
     * (LEFT, JUMP, SPEED) : [true, false, false, true, true]
     * (LEFT, JUMP, NOSPEED) : [true, false, false, false, true]
     * (DOWN, NOJUMP, SPEED) : [false, false, true, true, false]
     * (DOWN, NOJUMP, NOSPEED) : [false, false, true, false, false]
     * (NONE, NOJUMP, SPEED) : [false, false, false, true, false] **
     * (NONE, NOJUMP, NOSPEED) : [false, false, false, false, false]
     * (NONE, JUMP, SPEED) : [false, false, false, true, true] **
     * (NONE, JUMP, NOSPEED) : [false, false, false, false, true]

     */
    private final boolean[][] ACTION_MAP = {
            {false, true, false, true, false},
            {false, true, false, false, false},
            {false, true, false, true, true},
            {false, true, false, false, true},
            {true, false, false, true, false},
            {true, false, false, false, false},
            {true, false, false, true, true},
            {true, false, false, false, true},
            {false, false, true, true, false},
            {false, false, true, false, false},
            {false, false, false, true, false},
            {false, false, false, false, false},
            {false, false, false, true, true},
            {false, false, false, false, true}
    };

    public int prevAction = 0;

    public String prevState = null;

    private static final int CANVAS_MAX = 15;
    private static final int CANVAS_MIN = 0;

    public void setAlgorithm(String algorithm) {
        this.algorithm = algorithm;
    }

    public void setSelection(String selection) {
        this.selection = selection;
    }

    public void resetProgress(MarioForwardModel model) {
        this.percentCompleted = model.getCompletionPercentage();
        this.elimByStomp = 0;
        this.elimByFire = 0;
        this.bestProgress = percentCompleted;
        this.performanceData = new ArrayList<>();
    }

    public void initializeEpisode() {
        this.episodeData = new ArrayList<>();
    }

    private String velocityRepresentation(float[] velocity) {
        if (velocity[0] == 0 && velocity[1] == 0) {
            return "0000"; // Mario is staying in place
        } else if (velocity[0] > 0 && velocity[1] == 0) {
            return "0001"; // Mario is moving right
        } else if (velocity[0] > 0 && velocity[1] > 0) {
            return "0010"; // Mario is moving up and right
        } else if (velocity[0] > 0 && velocity[1] < 0) {
            return "0011"; // Mario is moving down and right
        } else if (velocity[0] < 0 && velocity[1] == 0) {
            return "0100"; // Mario is moving left
        } else if (velocity[0] < 0 && velocity[1] > 0) {
            return "0101"; // Mario is moving up and left
        } else if (velocity[0] < 0 && velocity[1] < 0) {
            return "0110"; // Mario is moving down and left
        } else if (velocity[0] == 0 && velocity[1] > 0) {
            return "0111"; // Mario is moving up
        } else {
            return "1000"; // Mario is moving down
        }
    }

    private String obstaclesAround(int[] marioPos, int[][] tiles) {
        boolean[] obstacles = new boolean[11];
        Arrays.fill(obstacles, false);
        // [HIGHEST, HIGH, LOW, LOWEST]
        int posX = marioPos[0] + 1;
        int posY = marioPos[1];

        for (int i=0;i<obstacles.length/2;i++) {
            if (posY + i > CANVAS_MAX) {
                break;
            }
            if (tiles[posX][posY+i-1] > 0) {
                obstacles[i] = true;
            }
        }
        for (int i=obstacles.length/2; i<obstacles.length-1;i++) {
            if (posY + i + 1 > CANVAS_MAX) {
                break;
            }

            if (tiles[posX+1][posY+i-1] > 0) {
                obstacles[i] = true;
            }
        }

//        if (this.selection.equals("epsilongreedytest")) {
//            System.out.println("*************************************************************************************************");
//            System.out.println(Arrays.toString(marioPos));
//            System.out.println("*************************************************************************************************");
//            for (int[] array : tiles) {
//                System.out.println(Arrays.toString(array));
//            }
//            System.out.println("*************************************************************************************************");
//        }
        if (marioPos[1] < CANVAS_MAX && tiles[marioPos[0]][marioPos[1]+1] > 0) {
            this.onTile = true;
            obstacles[obstacles.length-1] = true;
        } else {
            this.onTile = false;
            obstacles[obstacles.length-1] = false;
        }

        StringBuilder result = new StringBuilder(6);
        for (boolean obstacle : obstacles) {
            if (obstacle) {
                result.append('1');
            } else {
                result.append('0');
            }
        }
        return result.toString();
    }

    private String enemyDirections(int marioMode, int[] marioPos, int[][] enemies, int outerRange, int innerRange) {
        boolean[] enemyTable = new boolean[8];
        Arrays.fill(enemyTable, false);
        // [TOPLEFT, TOP, TOPRIGHT, LEFT, RIGHT, BOTTOMLEFT, BOTTOM, BOTTOMRIGHT]
        if (marioPos[0] < CANVAS_MIN || marioPos[0] > CANVAS_MAX || marioPos[1] < CANVAS_MIN || marioPos[1] > CANVAS_MAX) {
            return "00000000";
        }
        int leftOutX = marioPos[0] - outerRange/2;
        int leftX = marioPos[0] - innerRange/2;
        int rightOutX = marioPos[0] + outerRange/2;
        int rightX = marioPos[0] + innerRange/2;
        int topOutY = marioPos[1] - outerRange/2 - marioMode;
        int topY = marioPos[1] - innerRange/2 - marioMode;
        int bottomOutY = marioPos[1] + outerRange/2;
        int bottomY = marioPos[1] + innerRange/2;

//        System.out.println(outerRange + " " + innerRange);
//        System.out.println(marioPos[0] + " " + marioPos[1] + " " + leftOutX + " " + leftX + " " + rightOutX + " " + rightX + " " + topOutY + " " + topY + " " + bottomOutY + " " + bottomY);

        // TOP LEFT
        for (int i=Math.max(leftOutX, CANVAS_MIN);i<Math.min(leftX, CANVAS_MAX);i++) {
            for (int j=Math.max(topOutY, CANVAS_MIN);j<Math.min(topY, CANVAS_MAX);j++) {
                if (enemies[i][j] > 0) {
                    enemyTable[0] = true;
                }
            }
        }

        // TOP
        for (int j=Math.max(topOutY, CANVAS_MIN);j<Math.min(topY, CANVAS_MAX);j++) {
            if (enemies[marioPos[0]][j] > 0) {
                enemyTable[1] = true;
            }
        }

        // TOP RIGHT
        for (int i=Math.max(rightX+1, CANVAS_MIN);i<=Math.min(rightOutX, CANVAS_MAX);i++) {
            for (int j=Math.max(topOutY, CANVAS_MIN);j<Math.min(topY, CANVAS_MAX);j++) {
                if (enemies[i][j] > 0) {
                    enemyTable[2] = true;
                }
            }
        }

        // LEFT
        for (int i=Math.max(leftOutX, CANVAS_MIN);i<Math.min(leftX, CANVAS_MAX);i++) {
            if (enemies[i][marioPos[1]] > 0 || enemies[i][marioPos[1] - marioMode] > 0) {
                enemyTable[3] = true;
            }
        }


        // RIGHT
        for (int i=Math.max(rightX+1, CANVAS_MIN);i<=Math.min(rightOutX, CANVAS_MAX);i++) {
            if (enemies[i][marioPos[1]] > 0 || enemies[i][marioPos[1] - marioMode] > 0) {
                enemyTable[4] = true;
            }
        }

        // BOTTOM LEFT
        for (int i=Math.max(leftOutX, CANVAS_MIN);i<Math.min(leftX, CANVAS_MAX);i++) {
            for (int j=Math.max(bottomY+1, CANVAS_MIN);j<=Math.min(bottomOutY, CANVAS_MAX);j++) {
                if (enemies[i][j] > 0) {
                    enemyTable[5] = true;
                }
            }
        }

        // BOTTOM
        for (int j=Math.max(bottomY+1, CANVAS_MIN);j<=Math.min(bottomOutY, CANVAS_MAX);j++) {
            if (enemies[marioPos[0]][j] > 0) {
                enemyTable[6] = true;
            }
        }


        // BOTTOM RIGHT
        for (int i=Math.max(rightX+1, CANVAS_MIN);i<=Math.min(rightOutX, CANVAS_MAX);i++) {
            for (int j=Math.max(bottomY+1, CANVAS_MIN);j<=Math.min(bottomOutY, CANVAS_MAX);j++) {
                if (enemies[i][j] > 0) {
                    enemyTable[7] = true;
                }
            }
        }
        StringBuilder result = new StringBuilder(8);
        for (boolean enemy : enemyTable) {
            if (enemy) {
                result.append('1');
            } else {
                result.append('0');
            }
        }
        return result.toString();
    }

    public int greedySelection(String state) {
        if (!Q.containsKey(state)) {
            double[] stateActions = new double[ACTION_SPACE];
            Arrays.fill(stateActions, 0.0);
            Q.put(state, stateActions);
        }

        double[] stateActions = Q.get(state);

        int maxIndex = 0;
        double maxValue = stateActions[0];

        for (int i=1; i<stateActions.length;i++) {
            if (stateActions[i] > maxValue) {
                maxValue = stateActions[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public int epsilonGreedySelection(String state, double epsilon) {
        int maxIndex = 0;
        if (!this.algorithm.equals("doubleqlearning")) {
            if (!Q.containsKey(state)) {
                double[] stateActions = new double[ACTION_SPACE];
                Arrays.fill(stateActions, 0.0);
                Q.put(state, stateActions);
            }

            double[] stateActions = Q.get(state);

            double maxValue = stateActions[0];

            for (int i=1; i<stateActions.length;i++) {
                if (stateActions[i] > maxValue) {
                    maxValue = stateActions[i];
                    maxIndex = i;
                }
            }
        } else {
            if (!Q.containsKey(state)) {
                double[] stateActions = new double[ACTION_SPACE];
                Arrays.fill(stateActions, 0.0);
                Q.put(state, stateActions);
            }
            if (!Q2.containsKey(state)) {
                double[] stateActions = new double[ACTION_SPACE];
                Arrays.fill(stateActions, 0.0);
                Q2.put(state, stateActions);
            }

            double[] stateActions = Q.get(state);

            double[] stateActions2 = Q2.get(state);

            double maxValue = stateActions[0] + stateActions2[0];

            for (int i=1; i<stateActions.length;i++) {
                if (stateActions[i] + stateActions2[i] > maxValue) {
                    maxValue = stateActions[i] + stateActions2[i];
                    maxIndex = i;
                }
            }

        }

        Random rand = new Random();
        double randomDouble = rand.nextDouble();
        if (randomDouble <= epsilon) { // With probability of epsilon, select a random action
            maxIndex = rand.nextInt(ACTION_SPACE);
        }

        return maxIndex;

    }

    public void qlearningUpdate(String state, int action, double reward, String nextState) {
        double[] nextStateActions = Q.get(nextState);
        double maxValue = nextStateActions[0];
        for (int i=1;i<ACTION_SPACE;i++) {
            if (nextStateActions[i] > maxValue) {
                maxValue = nextStateActions[i];
            }
        }

        Q.get(state)[action] = ALPHA*(reward + GAMMA*maxValue - Q.get(state)[action]);
    }

    public void sarsaUpdate(String state, int action, double reward, String nextState, int nextAction) {
        Q.get(state)[action] = ALPHA*(reward + GAMMA*Q.get(nextState)[nextAction] - Q.get(state)[action]);
    }

    public void doubleqlearningUpdate(String state, int action, double reward, String nextState) {
        Random rand = new Random();

        int randomInt = rand.nextInt(2);

        if (randomInt == 0) {
            double[] nextStateActions = Q2.get(nextState);
            double maxValue = nextStateActions[0];
            for (int i=1;i<ACTION_SPACE;i++) {
                if (nextStateActions[i] > maxValue) {
                    maxValue = nextStateActions[i];
                }
            }

            Q.get(state)[action] = ALPHA*(reward + GAMMA*maxValue - Q.get(state)[action]);
        } else {
            double[] nextStateActions = Q.get(nextState);
            double maxValue = nextStateActions[0];
            for (int i=1;i<ACTION_SPACE;i++) {
                if (nextStateActions[i] > maxValue) {
                    maxValue = nextStateActions[i];
                }
            }

            Q2.get(state)[action] = ALPHA*(reward + GAMMA*maxValue - Q2.get(state)[action]);
        }
    }

    public void expectedsarsaUpdate(String state, int action, double reward, String nextState) {

        double[] nextStateActions = Q.get(nextState);
        int maxAction = 0;
        double maxValue = nextStateActions[0];
        for (int i=1;i<ACTION_SPACE;i++) {
            if (nextStateActions[i] > maxValue) {
                maxValue = nextStateActions[i];
                maxAction = i;
            }
        }

        double expectation = 0;

        expectation += (1 - EPSILON + EPSILON/(double)ACTION_SPACE)*maxValue;

        for (int i=0; i<ACTION_SPACE; i++) {
            if (i == maxAction) {
                continue;
            }
            expectation += (EPSILON/(double)ACTION_SPACE)*nextStateActions[i];
        }

        Q.get(state)[action] = ALPHA*(reward + GAMMA*expectation - Q.get(state)[action]);
    }

    @Override
    public void initialize(MarioForwardModel model, MarioTimer timer) {
        this.resetProgress(model);
        this.time = model.getRemainingTime();
        this.elevation = model.getMarioScreenTilePos()[1];
    }

    public void setupAgent() {
        /**
         * Actions
         * LEFT:0, RIGHT:1, DOWN:2, SPEED:3, JUMP:4
         */
        this.actions = new boolean[MarioActions.numberOfActions()];
        Arrays.fill(this.actions, false);
        this.Q = new HashMap<>();
        this.Q2 = new HashMap<>();
        this.performanceData = new ArrayList<>();
    }

    public String formulateState(MarioForwardModel model) {
        // Formulate state
        StringBuilder state = new StringBuilder(40);

        // 1. Mario Mode: 0 (small), 1 (big), 2 (fire)
        int marioMode = model.getMarioMode();

        if (marioMode == 0) {
            state.append("00");
        } else if (marioMode == 1) {
            state.append("01");
        } else {
            state.append("10");
        }

        // 2. Direction: The direction of Mario's velocity 9 possible values
        String velocityDir = velocityRepresentation(model.getMarioFloatVelocity());
        state.append(velocityDir);

        // 3. Stuck: A 0 or a 1 indicating if Mario has not moved for several frames (i.e., no movement is indicated by the direction attribute)
        if (velocityDir.equals("0000")) {
            this.stuckCounter++; // increment stuck counter
        } else {
            this.stuckCounter = 0; // reset stuck counter
        }

        boolean gotStuck = this.stuckCounter >= STUCK_LIMIT;
        if (gotStuck) {
            state.append('1');
        } else {
            state.append('0');
        }

        // 4. On Ground: A 0 or a 1
        boolean onGround = model.isMarioOnGround();
        if (onGround) {
            state.append('1');
        } else {
            state.append('0');
        }

        // 5. Can Jump: A 0 or a 1
        if (model.mayMarioJump()) {
            state.append('1');
        } else {
            state.append('0');
        }

        // 6. Collided with enemy
        boolean collided = marioMode < this.prevMarioMode;
        this.prevMarioMode = marioMode;
        if (collided) {
            state.append('1');
        } else {
            state.append('0');
        }

        int[] marioPos = model.getMarioScreenTilePos();

        // 7. Enemies Nearby: Indicates whether there is an enemy in the 8 directions within 8 (small) or 10 (large/fire) tiles surrounding Mario.
        // There is a 3x3 box for small Mario and a 4x3 box for large/fire Mario
        // 8. Enemies Midrange: Indicates whether there is an enemy in the 8 directions in the midrange around Mario (excludes close enemies).
        // There is a 7x7 box for small Mario and a 8x7 box for large/fire Mario. [8 bits]
        // 9. Enemies Far: Indicates whether there is an enemy in the 8 directions far from Mario (excludes midrange and close enemies).
        // There is a 11x11 box for small Mario and a 12x11 box for large/fire Mario. [8 bits]

        int[][] enemies = model.getMarioEnemiesObservation();

        // 7.
        String nearbyResult = enemyDirections(marioMode, marioPos, enemies, NEARBY_RADIUS, 0);
        state.append(nearbyResult);

        // 8.
        String midrangeResult = enemyDirections(marioMode, marioPos, enemies, MIDRANGE_RADIUS, NEARBY_RADIUS);
        state.append(midrangeResult);

        // 9.
        String farResult = enemyDirections(marioMode, marioPos, enemies, FARRANGE_RADIUS, MIDRANGE_RADIUS);
        state.append(farResult);

        // 10. Killed Enemy With Stomp: 0 or 1
        boolean byStomp = model.getKillsByStomp() > this.elimByStomp;
        this.elimByStomp = model.getKillsByStomp();
        if (byStomp) {
            state.append('1');
        } else {
            state.append('0');
        }

        // 11. Killed Enemy With Fire: 0 or 1
        boolean byFire = model.getKillsByFire() > this.elimByFire;
        this.elimByFire = model.getKillsByFire();
        if (byFire) {
            state.append('1');
        } else {
            state.append('0');
        }

        // 12. Obstacles: Indicates whether there exists obstacles in the 4 tiles in front of Mario.

        int[][] tiles = model.getMarioSceneObservation();
        tiles = model.getScreenSceneObservation();

        String obstaclesResult = obstaclesAround(marioPos, tiles);
        state.append(obstaclesResult);

        return state.toString();

        /*

            STATE: [MARIO MODE (2 bits)] [VELOCITY DIRECTION (4 bits)] [STUCK (1 bit)] [ON GROUND (1 bit)] [CAN JUMP (1 bit)]
            [COLLIDED (1 bit)] [NEARBY (8 bits) TL T TR L R BL B BR] [MIDRANGE (8 bits) TL T TR L R BL B BR] [FARRANGE (8 bits) TL T TR L R BL B BR]
            [BYSTOMP (1 bit)] [BYFIRE (1 bit)] [OBSTACLES (4 bits)]

         */
    }

    public double computeReward(MarioForwardModel model) {
        // Compute Reward
        double reward = 0;
        double progress = model.getCompletionPercentage();
        double difference = progress - this.percentCompleted;
        this.percentCompleted = progress;

        reward += PROGRESS_REWARD_MULTIPLIER*difference;

//        if (PROGRESS_REWARD_MULTIPLIER* difference < PROGRESS_TOL) {
//            reward -= 1;
//        }

        double deltaT = TIME_MULTIPLIER*(double)(model.getRemainingTime() - this.time )/100;

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
        }

        reward += ELEVATED_REWARD*elevationReward;

//        if (collided) {
//            reward -= COLLISION_PENALTY;
//        }

        int stompElim = 0;
        if (model.getKillsByStomp() > this.elimByStomp) {
            stompElim = model.getKillsByStomp() - this.elimByStomp;
            this.elimByStomp = model.getKillsByStomp();
        }
        reward += ELIM_REWARD*stompElim;

        int fireElim = 0;
        if (model.getKillsByFire() > this.elimByFire) {
            fireElim = model.getKillsByFire() - this.elimByFire;
            this.elimByFire = model.getKillsByFire();
        }
        reward += ELIM_REWARD*fireElim;

        if (model.getGameStatus().equals(GameStatus.WIN)) {
            reward += WIN_REWARD;
            System.out.println("**************************************************");
            System.out.println("VICTORY");
            System.out.println("**************************************************");
        } else if (model.getGameStatus().equals(GameStatus.LOSE)) {
            reward -= LOSE_PENALTY;
        }
        this.bestProgress = progress;
        return reward;
    }

    public int getNextAction(String state) {
        int nextAction = 0;
        if (this.selection.equals("epsilongreedy") || this.selection.equals("epsilongreedytest")) {
            nextAction = epsilonGreedySelection(state, EPSILON);
        } else if (this.selection.equals("greedy")) {
            nextAction = greedySelection(state);
        }
        return nextAction;
    }

    @Override
    public boolean[] getActions(MarioForwardModel model, MarioTimer timer) {

        if (model.getCompletionPercentage() > 0.355) {
            System.out.println(model.getCompletionPercentage() + " PASSED THE PIPES");
        }

        String state = formulateState(model);
        double reward = computeReward(model);
        int nextAction = getNextAction(state);

//        if (this.selection.equals("epsilongreedytest")) {
//            System.out.println(reward + " " + PROGRESS_REWARD_MULTIPLIER*difference + " " + deltaT + " " + elevationReward + " " + stompElim + " " + fireElim + " " + model.getGameStatus());
//        }


        // Perform update
        if (this.prevState == null) {
            this.prevState = state;
        }
        if (this.algorithm.equals("qlearning")) {
            qlearningUpdate(prevState, prevAction, reward, state);
        } else if (this.algorithm.equals("sarsa")) {
            sarsaUpdate(prevState, prevAction, reward, state, nextAction);
        } else if (this.algorithm.equals("doubleqlearning")) {
            doubleqlearningUpdate(prevState, prevAction, reward, state);
        } else if (this.algorithm.equals("expectedsarsa")) {
            expectedsarsaUpdate(prevState, prevAction, reward, state);
        }

        this.prevAction = nextAction;
        this.prevState = state;

        String[] data = new String[3];

        data[0] = String.valueOf(model.getCompletionPercentage());
        data[1] = model.getGameStatus().toString();
        data[2] = String.valueOf(((double)(400000 - model.getRemainingTime())/1000));

        episodeData.add(data);

        if (model.getGameStatus().equals(GameStatus.LOSE) || model.getGameStatus().equals(GameStatus.TIME_OUT) || model.getGameStatus().equals(GameStatus.WIN)) {
            performanceData.add(episodeData);
        }

        return ACTION_MAP[nextAction].clone();
    }

    @Override
    public String getAgentName() {
        return "Super Mario RL";
    }
}
