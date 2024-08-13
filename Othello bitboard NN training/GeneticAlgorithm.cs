using Othello_bitboard_NN_training;
using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Othello_bitboard_NN_training
{
    public class GeneticAlgorithm
    {
        private Random random;
        private int populationSize;
        private double mutationRate;
        private double crossoverRate;
        private int gamesPerPair;

        private readonly object progressLock = new object();

        public GeneticAlgorithm(int populationSize, double mutationRate, double crossoverRate, int gamesPerPair)
        {
            this.random = new Random();
            this.populationSize = populationSize;
            this.mutationRate = mutationRate;
            this.crossoverRate = crossoverRate;
            this.gamesPerPair = gamesPerPair;
        }

        public List<Bot> Evolve(List<Bot> evaluatedPopulation, int generation)
        {
            // Sort the population by fitness (descending order)
            var sortedPopulation = evaluatedPopulation.OrderByDescending(bot => bot.WinRate).ToList();

            // Retain the top 10% bots without mutation (elite)
            int eliteCount = (int)(populationSize * 0.1);
            List<Bot> newPopulation = sortedPopulation.Take(eliteCount).Select(bot => bot).ToList();

            // Mutate the next 40%
            int mutateCount = (int)(populationSize * 0.4);
            var toMutate = sortedPopulation.Skip(eliteCount).Take(mutateCount).ToList();
            foreach (var bot in toMutate)
            {
                Mutate(bot.GetNeuralNetwork());
                newPopulation.Add(bot);
            }

            // Apply softmax to the top 50% bots for parent selection
            int topHalfCount = (int)(populationSize * 0.5);
            var topHalfPopulation = sortedPopulation.Take(topHalfCount).ToList();
            double[] fitnessScores = topHalfPopulation.Select(bot => bot.WinRate).ToArray();
            double[] softmaxScores = Softmax(fitnessScores);

            // Fill the rest of the population aka the bottom 50%
            while (newPopulation.Count < populationSize)
            {
                // Select parents based on softmax probabilities
                Bot parent1 = topHalfPopulation[SelectByProbability(softmaxScores)];
                Bot parent2 = topHalfPopulation[SelectByProbability(softmaxScores)];

                NeuralNetwork childNN = Crossover(parent1.GetNeuralNetwork(), parent2.GetNeuralNetwork());
                Mutate(childNN);

                // Create the new bot with the correct naming
                Bot childBot = new Bot(childNN)
                {
                    Name = $"Gen{generation + 1}_Bot{newPopulation.Count + 1}",
                    Parents = $"{parent1.Name} / {parent2.Name}"
                };

                newPopulation.Add(childBot);
            }

            return newPopulation;
        }

        private NeuralNetwork Crossover(NeuralNetwork parent1, NeuralNetwork parent2)
        {
            NeuralNetwork child = new NeuralNetwork(parent1.InputSize, parent1.HiddenSize, parent1.OutputSize);

            for (int i = 0; i < child.InputWeights.Length; i++)
            {
                child.InputWeights[i] = random.NextDouble() < crossoverRate ? parent1.InputWeights[i] : parent2.InputWeights[i];
            }

            for (int i = 0; i < child.HiddenWeights.Length; i++)
            {
                child.HiddenWeights[i] = random.NextDouble() < crossoverRate ? parent1.HiddenWeights[i] : parent2.HiddenWeights[i];
            }

            return child;
        }

        private void Mutate(NeuralNetwork nn)
        {
            for (int i = 0; i < nn.InputWeights.Length; i++)
            {
                if (random.NextDouble() < mutationRate)
                {
                    nn.InputWeights[i] += (random.NextDouble() * 2 - 1) * 0.1; // Mutate by a small random value
                }
            }

            for (int i = 0; i < nn.HiddenWeights.Length; i++)
            {
                if (random.NextDouble() < mutationRate)
                {
                    nn.HiddenWeights[i] += (random.NextDouble() * 2 - 1) * 0.1; // Mutate by a small random value
                }
            }
        }

        public List<Bot> EvaluatePopulation(List<Bot> population, int generation, int maxThreads = 12)
        {
            List<Bot> evaluatedPopulation = new();
            var concurrentScoreCounts = new ConcurrentDictionary<Bot, double>();

            // Initialize win counts to zero
            foreach (var bot in population)
            {
                concurrentScoreCounts[bot] = 0;
            }

            // Calculate the number of games that will be played
            int totalNumberOfGames = Enumerable.Range(1, population.Count - 1).Sum(i => i) * 2;
            int gamesPlayed = 0;

            UpdateProgress(gamesPlayed, totalNumberOfGames, generation);

            // Set up ParallelOptions to limit the number of threads
            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = maxThreads
            };

            // Parallel loop to evaluate bots with controlled number of threads
            Parallel.For(0, population.Count, parallelOptions, i =>
            {
                for (int j = i + 1; j < population.Count; j++)
                {
                    var bot1 = population[i];
                    var bot2 = population[j];

                    for (int k = 0; k < gamesPerPair; k++)
                    {
                        // Play the game and update win counts and ratings
                        SimulateGame(bot1, bot2, concurrentScoreCounts);
                        SimulateGame(bot2, bot1, concurrentScoreCounts);

                        gamesPlayed += 2;

                        lock (progressLock)
                        {
                            UpdateProgress(gamesPlayed, totalNumberOfGames, generation);
                        }
                    }
                }
            });

            // Convert win counts back to evaluated population
            foreach (var bot in population)
            {
                double winRate = 100 * concurrentScoreCounts[bot] / (gamesPerPair * 2 * (population.Count - 1));
                bot.WinRate = winRate;
                evaluatedPopulation.Add(bot);
            }

            return evaluatedPopulation;
        }

        private void SimulateGame(Bot bot1, Bot bot2, ConcurrentDictionary<Bot, double> concurrentWinCounts)
        {
            // Clone bots to ensure independent game state
            Bot bot1Clone = new(bot1.GetNeuralNetwork().Clone());
            Bot bot2Clone = new(bot2.GetNeuralNetwork().Clone());

            int result = bot1Clone.PlayGameAgainst(bot2Clone);
            AdjustRatings(bot1, bot2, result);

            if (result == 1)
            {
                concurrentWinCounts.AddOrUpdate(bot1, 1, (key, oldValue) => oldValue + 1);
            }
            else if (result == -1)
            {
                concurrentWinCounts.AddOrUpdate(bot2, 1, (key, oldValue) => oldValue + 1);
            }
            else
            {
                concurrentWinCounts.AddOrUpdate(bot1, 0.5, (key, oldValue) => oldValue + 0.5);
                concurrentWinCounts.AddOrUpdate(bot2, 0.5, (key, oldValue) => oldValue + 0.5);
            }
        }

        public void UpdateProgress(int gamesPlayed, int totalNumberOfGames, int generation)
        {
            double progress = (double) 100 * gamesPlayed / totalNumberOfGames;
            Console.Write($"\rProgress on generation {generation}: {Math.Round(progress, 0)} %");
        }

        public void DisplayBotsWithWinRates(List<Bot> evaluatedPopulation, int gen)
        {
            // Sort the evaluated population by win rate in descending order
            var sortedPopulation = evaluatedPopulation.OrderByDescending(bot => bot.WinRate).ToList();

            Console.WriteLine("\n");

            // Display each bot with its win rate
            foreach (var bot in sortedPopulation)
            {
                if (bot.Parents != "")
                {
                    Console.WriteLine($"{bot.Name}, Win Rate: {Math.Round(bot.WinRate, 0)} %, Rating: {bot.Rating}, Parents: {bot.Parents}");
                }
                else
                {
                    Console.WriteLine($"{bot.Name}, Win Rate: {Math.Round(bot.WinRate, 0)} %, Rating: {bot.Rating}");
                }
            }
        }

        public void TestBestBotAgainstBenchmark(Bot bestBot, BenchmarkBot benchmarkBot) // Multi thread this!
        {
            double totalScore = 0;

            int numberOfGames = 10; // Play multiple games to get a reliable result

            for (int i = 0; i < numberOfGames; i++)
            {
                // Best bot plays black
                int result = benchmarkBot.PlayGame(bestBot, true);
                if (result == 1) totalScore++;
                else if (result == 0) totalScore += 0.5;

                // Best bot plays white
                result = benchmarkBot.PlayGame(bestBot, false);
                if (result == -1) totalScore++;
                else if (result == 0) totalScore += 0.5;
            }
            double winrate = 100.0 * totalScore / (numberOfGames * 2);
            Console.WriteLine($"\n{bestBot.Name} vs. Benchmark: Winrate: {Math.Round(winrate, 0)} %");
        }

        private double[] Softmax(double[] values)
        {
            double temperature = 2.0; // The temperature parameter controls the "sharpness" of the output probabilities
            double max = values.Max();
            double sumExp = values.Sum(v => Math.Exp((v - max) / temperature));
            return values.Select(v => Math.Exp((v - max) / temperature) / sumExp).ToArray();
        }

        private int SelectByProbability(double[] probabilities)
        {
            double cumulative = 0.0;
            double randomValue = random.NextDouble();

            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                if (randomValue <= cumulative)
                {
                    return i;
                }
            }
            return probabilities.Length - 1; // Return the last index in case of rounding issues
        }


        public void AdjustRatings(Bot bot1, Bot bot2, int result)
        {
            // Determine the K-factor for each bot
            int K = 32;

            // Calculate the expected score for each bot
            double expectedScore1 = CalculateExpectedScore(bot1.Rating, bot2.Rating);
            double expectedScore2 = CalculateExpectedScore(bot2.Rating, bot1.Rating);

            // Determine the actual score based on the result
            double actualScore1, actualScore2;

            if (result == 1) // bot1 wins
            {
                actualScore1 = 1.0;
                actualScore2 = 0.0;
            }
            else if (result == -1) // bot2 wins
            {
                actualScore1 = 0.0;
                actualScore2 = 1.0;
            }
            else // draw
            {
                actualScore1 = 0.5;
                actualScore2 = 0.5;
            }

            // Adjust the ratings
            bot1.Rating = (int)Math.Round(bot1.Rating + K * (actualScore1 - expectedScore1));
            bot2.Rating = (int)Math.Round(bot2.Rating + K * (actualScore2 - expectedScore2));
        }

        private double CalculateExpectedScore(int rating1, int rating2)
        {
            return 1.0 / (1.0 + Math.Pow(10, (rating2 - rating1) / 400.0));
        }
    }
}
