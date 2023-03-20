import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import agents.superMarioRL.Agent;
import engine.core.MarioAgent;
import engine.core.MarioGame;
import engine.core.MarioResult;

public class PlayLevel {

    public static final int runs = 10;
    public static final int episodes = 150000;
    public static void printResults(MarioResult result) {
        System.out.println("****************************************************************");
        System.out.println("Game Status: " + result.getGameStatus().toString() +
                " Percentage Completion: " + result.getCompletionPercentage());
        System.out.println("Lives: " + result.getCurrentLives() + " Coins: " + result.getCurrentCoins() +
                " Remaining Time: " + (int) Math.ceil(result.getRemainingTime() / 1000f));
        System.out.println("Mario State: " + result.getMarioMode() +
                " (Mushrooms: " + result.getNumCollectedMushrooms() + " Fire Flowers: " + result.getNumCollectedFireflower() + ")");
        System.out.println("Total Kills: " + result.getKillsTotal() + " (Stomps: " + result.getKillsByStomp() +
                " Fireballs: " + result.getKillsByFire() + " Shells: " + result.getKillsByShell() +
                " Falls: " + result.getKillsByFall() + ")");
        System.out.println("Bricks: " + result.getNumDestroyedBricks() + " Jumps: " + result.getNumJumps() +
                " Max X Jump: " + result.getMaxXJump() + " Max Air Time: " + result.getMaxJumpAirTime());
        System.out.println("****************************************************************");
    }

    public static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
        } catch (IOException e) {
        }
        return content;
    }

    public static void train(String level) {
        MarioGame game = new MarioGame();

        for (int i=0; i<runs;i++) {

            Agent agent = new agents.superMarioRL.Agent();
            setupAgentAndGame(agent, game, "qlearning", "epsilongreedy");
            for (int j=0; j<episodes; j++) {
                if (j >= 999 && (j-999)%1000 == 0) {
                    agent.setSelection("greedy");
                    game.setupVisuals(4);
                    printResults(game.runGame(agent, getLevel(level), 30, 0, true, 30, 4));
//                    game.gameLoop(getLevel(level), 20, 0, true, 30);
                    agent.setSelection("epsilongreedy");
                }
                System.out.println("Episode: " + (j+1));
                game.runGame(agent, getLevel(level), 30, 0, false, 0, 4);
//                game.gameLoop(getLevel(level), 20, 0, false, 0);
            }
        }

    }
    public static void setupAgentAndGame(Agent agent, MarioGame game, String algorithm, String selection) {
        agent.setAlgorithm(algorithm);
        agent.setSelection(selection);
        agent.setupAgent();
        game.setAgent(agent);
    }
    public static void main(String[] args) {
//        MarioGame game = new MarioGame();
//        Agent agent = new agents.superMarioRL.Agent();
//        setupAgentAndGame(agent, game, "qlearning", "epsilongreedy");
//        printResults(game.runGame(agent, getLevel("./levels/original/lvl-1.txt"), 20, 0, true, 30, 4));
        train("./levels/original/lvl-1.txt");
    }
}
