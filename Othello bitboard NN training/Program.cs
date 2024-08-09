using Othello_bitboard_NN_training;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Othello_Bitboard_CA
{
    class Program
    {
        static void Main(string[] args)
        {
            EvolutionSimulation simulation = new EvolutionSimulation();
            simulation.Run();
        }
    }
}
