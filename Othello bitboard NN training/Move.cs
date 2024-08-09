using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Othello_bitboard_NN_training
{
    public class Move
    {
        public int Position { get; set; }
        public double Evaluation { get; set; }
        public int DepthReached { get; set; }

        public Move(int position, double evaluation, int depthReached)
        {
            Position = position;
            Evaluation = evaluation;
            DepthReached = depthReached;
        }
    }

}
