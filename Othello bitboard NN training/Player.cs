using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Othello_bitboard_NN_training
{
    public struct Player
    {
        public bool IsBlack { get; }
        public bool IsMaximizing { get; }

        public Player(bool isBlack, bool isMaximizing)
        {
            IsBlack = isBlack;
            IsMaximizing = isMaximizing;
        }

        public Player GetOpponent()
        {
            return new Player(!IsBlack, !IsMaximizing);
        }
    }
}
