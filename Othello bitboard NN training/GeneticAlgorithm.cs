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

        public List<Bot> Evolve(List<Tuple<Bot, double>> evaluatedPopulation, int generation)
        {
            // Sort the population by fitness (descending order)
            var sortedPopulation = evaluatedPopulation.OrderByDescending(tuple => tuple.Item2).ToList();

            // Select the top 30% to carry over
            int topCount = (int)(populationSize * 0.3);
            List<Bot> newPopulation = sortedPopulation.Take(topCount).Select(tuple => tuple.Item1).ToList();

            // Fill the rest of the population with offspring generated through crossover and mutation
            while (newPopulation.Count < populationSize)
            {
                Bot parent1 = newPopulation[random.Next(topCount)];
                Bot parent2 = newPopulation[random.Next(topCount)];

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

        public List<Tuple<Bot, double>> EvaluatePopulation(List<Bot> population, int generation)
        {
            List<Tuple<Bot, double>> evaluatedPopulation = new List<Tuple<Bot, double>>();
            var concurrentWinCounts = new ConcurrentDictionary<Bot, double>();

            // Initialize win counts to zero
            foreach (var bot in population)
            {
                concurrentWinCounts[bot] = 0;
            }

            // Calculate the number of games that will be played
            int totalNumberOfGames = Enumerable.Range(1, population.Count - 1).Sum(i => i) * 2;
            int gamesPlayed = 0;

            UpdateProgress(gamesPlayed, totalNumberOfGames, generation);

            // Parallel loop to evaluate bots
            Parallel.For(0, population.Count, i =>
            {
                for (int j = i + 1; j < population.Count; j++)
                {
                    var bot1 = population[i];
                    var bot2 = population[j];

                    for (int k = 0; k < gamesPerPair; k++)
                    {
                        // Play the game and update win counts and ratings
                        SimulateGame(bot1, bot2, concurrentWinCounts);
                        SimulateGame(bot2, bot1, concurrentWinCounts);

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
                double winRate = concurrentWinCounts[bot] / (gamesPerPair * 2 * (population.Count - 1));
                evaluatedPopulation.Add(new Tuple<Bot, double>(bot, winRate));
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
            double progress = (double)gamesPlayed / totalNumberOfGames;
            Console.Write($"\rProgress on generation {generation}: {Math.Round(progress * 100, 0)} %");
        }

        public void DisplayBotsWithWinRates(List<Tuple<Bot, double>> evaluatedPopulation, int gen)
        {
            // Sort the evaluated population by win rate in descending order
            var sortedPopulation = evaluatedPopulation.OrderByDescending(tuple => tuple.Item2).ToList();

            Console.WriteLine("\n");

            // Display each bot with its win rate
            foreach (var botTuple in sortedPopulation)
            {
                if (botTuple.Item1.Parents != "")
                {
                    Console.WriteLine($"{botTuple.Item1.Name}, Win Rate: {botTuple.Item2:F2}, Rating: {botTuple.Item1.Rating}, Parents: {botTuple.Item1.Parents}");
                }
                else
                {
                    Console.WriteLine($"{botTuple.Item1.Name}, Win Rate: {botTuple.Item2:F2}, Rating: {botTuple.Item1.Rating}");
                }
            }
        }

        public void TestBestBotAgainstBenchmark(Bot bestBot, BenchmarkBot benchmarkBot)
        {
            int bestBotWins = 0;
            int benchmarkBotWins = 0;
            int draws = 0;

            int numberOfGames = 25; // Play multiple games to get a reliable result

            for (int i = 0; i < numberOfGames; i++)
            {
                // Best bot plays black
                int result = benchmarkBot.PlayGame(bestBot, true);
                if (result == 1) bestBotWins++;
                else if (result == -1) benchmarkBotWins++;
                else draws++;

                // Best bot plays white
                result = benchmarkBot.PlayGame(bestBot, false);
                if (result == -1) bestBotWins++;
                else if (result == 1) benchmarkBotWins++;
                else draws++;
            }

            Console.WriteLine($"\n{bestBot.Name} vs. Benchmark: {bestBotWins} Wins, {benchmarkBotWins} Losses, {draws} Draws");
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
