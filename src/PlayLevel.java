import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import agents.superMarioRL.Agent;
import com.opencsv.CSVWriter;
import engine.core.MarioGame;
import engine.core.MarioResult;

public class PlayLevel {

    public static final int runs = 10;
    public static final int episodes = 10000;
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

    public static void train(String level) throws IOException {
        MarioGame game = new MarioGame();

        for (int i=0; i<runs;i++) {

            Agent agent = new agents.superMarioRL.Agent();
            setupAgentAndGame(agent, game, "doubleqlearning", "epsilongreedy");
            for (int j=0; j<episodes; j++) {
//                if (j%2500 == 0) {
//                    agent.setSelection("epsilongreedytest");
//                    game.setupVisuals(4);
//                    printResults(game.runGame(agent, getLevel(level), 60, 0, true, 30, 4));
//                    agent.setSelection("epsilongreedy");
//                }
                System.out.println("Episode: " + (j+1));
                game.runGame(agent, getLevel(level), 400, 0, false, 0, 4);
            }
            ArrayList<String[]> data = agent.getPerformanceData();
            try {
                File myObj = new File("./data/doubleqlearning_run"+(i+1)+".csv");
                if (myObj.createNewFile()) {
                    System.out.println("File created: " + myObj.getName());
                } else {
                    System.out.println("File already exists.");
                }
            } catch (IOException e) {
                System.out.println("An error occurred.");
                e.printStackTrace();
            }
            try (CSVWriter writer = new CSVWriter(new FileWriter("./data/doubleqlearning_run"+(i+1)+".csv"))) {
                for (String[] episode: data) {
                    writer.writeNext(episode);
                }
            }
        }

    }
    public static void setupAgentAndGame(Agent agent, MarioGame game, String algorithm, String selection) {
        agent.setAlgorithm(algorithm);
        agent.setSelection(selection);
        agent.setupAgent();
        game.setAgent(agent);
    }
    public static void main(String[] args) throws IOException {
//        MarioGame game = new MarioGame();
//        Agent agent = new agents.superMarioRL.Agent();
//        setupAgentAndGame(agent, game, "qlearning", "epsilongreedy");
//        printResults(game.runGame(agent, getLevel("./levels/original/lvl-1.txt"), 20, 0, true, 30, 4));
        train("./levels/original/lvl-1.txt");
    }
}
