using Othello_bitboard_NN_training;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Othello_bitboard_NN_training
{
    public class NeuralNetwork
    {
        public int InputSize { get; private set; }
        public int HiddenSize { get; private set; }
        public int OutputSize { get; private set; }
        private readonly Random random;

        public double[] InputWeights { get; private set; }
        public double[] HiddenWeights { get; private set; }

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        {
            InputSize = inputSize;
            HiddenSize = hiddenSize;
            OutputSize = outputSize;
            this.random = new Random();

            InputWeights = new double[InputSize * HiddenSize];
            HiddenWeights = new double[HiddenSize * OutputSize];

            InitializeWeights();
        }

        private void InitializeWeights()
        {
            for (int i = 0; i < InputWeights.Length; i++)
            {
                InputWeights[i] = random.NextDouble() * 2 - 1; // Initialize weights between -1 and 1
            }

            for (int i = 0; i < HiddenWeights.Length; i++)
            {
                HiddenWeights[i] = random.NextDouble() * 2 - 1; // Initialize weights between -1 and 1
            }
        }

        public NeuralNetwork Clone()
        {
            NeuralNetwork clone = new NeuralNetwork(InputSize, HiddenSize, OutputSize);
            Array.Copy(this.InputWeights, clone.InputWeights, this.InputWeights.Length);
            Array.Copy(this.HiddenWeights, clone.HiddenWeights, this.HiddenWeights.Length);
            return clone;
        }

        public double Evaluate(double[] inputs)
        {
            if (inputs.Length != InputSize)
            {
                throw new ArgumentException($"Input size must be {InputSize}");
            }

            double[] hiddenLayerOutputs = new double[HiddenSize];

            // Compute hidden layer activations using ReLU
            for (int i = 0; i < HiddenSize; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < InputSize; j++)
                {
                    sum += inputs[j] * InputWeights[j + i * InputSize];
                }
                hiddenLayerOutputs[i] = ReLU(sum);
            }

            // Compute output layer activations and apply tanh to ensure output is between -1 and 1
            double output = 0.0;
            for (int i = 0; i < HiddenSize; i++)
            {
                output += hiddenLayerOutputs[i] * HiddenWeights[i];
            }

            return Math.Tanh(output);
        }

        // ReLU activation function
        private double ReLU(double x)
        {
            return Math.Max(0, x);
        }
        

        public void SaveWeights(string filePath)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine(string.Join(",", InputWeights));
                writer.WriteLine(string.Join(",", HiddenWeights));
            }
        }

        public void LoadWeights(string filePath)
        {
            using (StreamReader reader = new StreamReader(filePath))
            {
                InputWeights = reader.ReadLine().Split(',').Select(double.Parse).ToArray();
                HiddenWeights = reader.ReadLine().Split(',').Select(double.Parse).ToArray();
            }
        }
    }

}
