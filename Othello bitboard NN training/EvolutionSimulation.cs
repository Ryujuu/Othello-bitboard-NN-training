using Othello_bitboard_NN_training;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Othello_bitboard_NN_training
{
    public class EvolutionSimulation
    {
        public void Run()
        {
            int inputSize = 8; // Number of features
            int hiddenSize = 16; // Size of the hidden layer
            int outputSize = 1; // Single output for evaluation score

            int populationSize = 50;
            double mutationRate = 0.01;
            double crossoverRate = 0.7;
            int generations = 100;
            int gamesPerPair = 1; // Number of games played as black and white to simulate per pair of neural networks

            List<Bot> population = new List<Bot>();
            for (int i = 0; i < populationSize; i++)
            {
                NeuralNetwork nn = new NeuralNetwork(inputSize, hiddenSize, outputSize);
                Bot bot = new Bot(nn);

                // Check if weights file exists and load weights
                string weightsFilePath = "best_bot_weights.csv";
                if (File.Exists(weightsFilePath) && i == 0) // Load weights into the first bot if the file exists
                {
                    nn.LoadWeights(weightsFilePath);
                    bot.Name = "Gen1_Loaded";
                }
                else
                {
                    bot.Name = $"Gen1_Bot{i + 1}";
                }
                population.Add(bot);
            }

            GeneticAlgorithm ga = new GeneticAlgorithm(populationSize, mutationRate, crossoverRate, gamesPerPair);
            BenchmarkBot benchmarkBot = new BenchmarkBot();

            for (int gen = 1; gen <= generations; gen++)
            {
                List<Tuple<Bot, double>> evaluatedPopulation = ga.EvaluatePopulation(population, gen);
                ga.DisplayBotsWithWinRates(evaluatedPopulation, gen);

                // Sort the evaluated population by fitness (descending order)
                evaluatedPopulation = evaluatedPopulation.OrderByDescending(tuple => tuple.Item2).ToList();

                // Select the best bot (the one with the highest fitness)
                Bot bestBot = evaluatedPopulation.First().Item1;

                // Save the best bot's neural network weights
                bestBot.GetNeuralNetwork().SaveWeights($"best_bot_weights.csv");
                // Test the best bot against the benchmark bot
                ga.TestBestBotAgainstBenchmark(bestBot, benchmarkBot);

                population = ga.Evolve(evaluatedPopulation, gen);
                Console.WriteLine($"\nGeneration {gen} evolved.\n");
            }
        }
    }
}
