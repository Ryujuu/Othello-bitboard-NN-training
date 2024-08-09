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

        public List<Bot> Evolve(List<Tuple<Bot, double>> population, int generation)
        {
            // Select the top 30% as parents
            List<Bot> parents = SelectParents(population);

            // Generate new population through crossover and mutation
            List<Bot> newPopulation = new List<Bot>();
            while (newPopulation.Count < populationSize)
            {
                Bot parent1 = parents[random.Next(parents.Count)];
                Bot parent2 = parents[random.Next(parents.Count)];

                NeuralNetwork childNN = Crossover(parent1.GetNeuralNetwork(), parent2.GetNeuralNetwork());
                Mutate(childNN);

                // Create the new bot with the correct naming
                Bot childBot = new Bot(childNN)
                {
                    Name = $"Gen{generation}_Bot{newPopulation.Count}",
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
                        // Clone bots to ensure independent game state
                        var bot1Clone = new Bot(bot1.GetNeuralNetwork().Clone());
                        var bot2Clone = new Bot(bot2.GetNeuralNetwork().Clone());

                        int result = bot1Clone.PlayGameAgainst(bot2Clone);
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

                        // Play the game in the opposite direction
                        result = bot2Clone.PlayGameAgainst(bot1Clone);
                        if (result == -1)
                        {
                            concurrentWinCounts.AddOrUpdate(bot1, 1, (key, oldValue) => oldValue + 1);
                        }
                        else if (result == 1)
                        {
                            concurrentWinCounts.AddOrUpdate(bot2, 1, (key, oldValue) => oldValue + 1);
                        }
                        else
                        {
                            concurrentWinCounts.AddOrUpdate(bot1, 0.5, (key, oldValue) => oldValue + 0.5);
                            concurrentWinCounts.AddOrUpdate(bot2, 0.5, (key, oldValue) => oldValue + 0.5);
                        }
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
                if (gen > 1)
                {
                    Console.WriteLine($"{botTuple.Item1.Name} from {botTuple.Item1.Parents}, Win Rate: {botTuple.Item2:F2}");
                }
                else
                {
                    Console.WriteLine($"{botTuple.Item1.Name}, Win Rate: {botTuple.Item2:F2}");
                }
            }
        }

        private List<Bot> SelectParents(List<Tuple<Bot, double>> evaluatedPopulation)
        {
            return evaluatedPopulation
                .OrderByDescending(tuple => tuple.Item2)
                .Take((int)(populationSize * 0.3))
                .Select(tuple => tuple.Item1)
                .ToList();
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
    }
}
