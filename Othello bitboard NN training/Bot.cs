using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Othello_bitboard_NN_training
{

    public class Bot
    {
        private int FixedDepth = 6;
        private NeuralNetwork NN;
        private BoardFeatures BoardFeatures;

        public string Name { get; set; }
        public string Parents { get; set; }

        public int Rating { get; set; }
        public Bot(NeuralNetwork nn)
        {
            NN = nn;
            BoardFeatures = new BoardFeatures();
            Rating = 1000;
            Parents = "";
        }

        public NeuralNetwork GetNeuralNetwork()
        {
            return NN;
        }

        public Move GetMove(Board board, Player currentPlayer)
        {
            Move bestMove;

            bestMove = Search(board.Clone(), FixedDepth, currentPlayer, double.MinValue, double.MaxValue);
            
            return bestMove;
        }

        private Move Search(Board board, int depth, Player currentPlayer, double alpha, double beta)
        {
            if (depth == 0 || board.IsGameOver())
            {
                double evaluation = EvaluateBoard(board, currentPlayer);
                return new Move(-1, evaluation, depth);
            }

            ulong validMoves = board.GenerateValidMoves(currentPlayer.IsBlack);
            if (validMoves == 0)
            {
                return Search(board, depth - 1, currentPlayer.GetOpponent(), alpha, beta);
            }

            int bestMove = -1;
            double bestScore = currentPlayer.IsMaximizing ? double.MinValue : double.MaxValue;

            for (int i = 0; i < 64; i++)
            {
                if ((validMoves & (1UL << i)) != 0)
                {
                    Board clonedBoard = board.Clone();
                    clonedBoard.MakeMove(i, currentPlayer.IsBlack);

                    Move moveEval = Search(clonedBoard, depth - 1, currentPlayer.GetOpponent(), alpha, beta);

                    if (currentPlayer.IsMaximizing)
                    {
                        if (moveEval.Evaluation > bestScore)
                        {
                            bestScore = moveEval.Evaluation;
                            bestMove = i;
                        }
                        alpha = Math.Max(alpha, bestScore);
                    }
                    else
                    {
                        if (moveEval.Evaluation < bestScore)
                        {
                            bestScore = moveEval.Evaluation;
                            bestMove = i;
                        }
                        beta = Math.Min(beta, bestScore);
                    }

                    if (beta <= alpha)
                    {
                        break; // Beta cut-off
                    }
                }
            }

            return new Move(bestMove, bestScore, depth);
        }

        private double EvaluateBoard(Board board, Player currentPlayer)
        {
            var features = BoardFeatures.GetBoardFeatures(board, currentPlayer.IsBlack).ToArray();
            return NN.Evaluate(features);
        }

        public int PlayGameAgainst(Bot opponent)
        {
            Board board = new Board();
            Player currentPlayer = new Player(true, true);
            Player opponentPlayer = currentPlayer.GetOpponent();
            bool isBlack = true;

            while (!board.IsGameOver())
            {
                Move move = null;
                bool hasValidMoves = board.GenerateValidMoves(isBlack) != 0;

                if (hasValidMoves)
                {
                    move = isBlack ? GetMove(board, currentPlayer) : opponent.GetMove(board, opponentPlayer);

                    if (move.Position != -1)
                    {
                        board.MakeMove(move.Position, isBlack);
                    }
                }
                // Switch players
                isBlack = !isBlack;
            }

            int blackDiscs = board.CountBits(board.BlackBoard);
            int whiteDiscs = board.CountBits(board.WhiteBoard);

            // Display board after each game to see if it works as expected
            //PrintBoard(board);

            if (blackDiscs > whiteDiscs)
                return isBlack ? 1 : -1; // Current bot wins
            else if (whiteDiscs > blackDiscs)
                return isBlack ? -1 : 1; // Opponent bot wins
            else
                return 0; // Draw
        }


        // Print the current board state
        // ANSI escape codes for coloring
        private const string Reset = "\x1b[0m";
        private const string DarkGreen = "\x1b[48;5;22m"; // Dark green background
        private const string WhitePiece = "\x1b[97m"; // White text
        private const string BlackPiece = "\x1b[30m"; // Black text
        public static void PrintBoard(Board board)
        {
            Console.WriteLine("    A   B   C   D   E   F   G   H  ");
            Console.WriteLine($"  {DarkGreen}+---+---+---+---+---+---+---+---+{Reset}");

            for (int row = 7; row >= 0; row--) // Start from row 7 (8 in display) to row 0 (1 in display)
            {
                Console.Write(row + 1 + $" {DarkGreen}|{Reset}");
                for (int col = 0; col < 8; col++)
                {
                    int position = row * 8 + col;
                    string piece = GetPieceAtPosition(board, position);
                    Console.Write($"{DarkGreen} {piece} {Reset}{DarkGreen}|{Reset}");
                }
                Console.WriteLine(" " + (row + 1));
                Console.WriteLine($"  {DarkGreen}+---+---+---+---+---+---+---+---+{Reset}");
            }
            Console.WriteLine("    A   B   C   D   E   F   G   H  ");
        }

        private static string GetPieceAtPosition(Board board, int position)
        {
            if ((board.BlackBoard & (1UL << position)) != 0) return BlackPiece + "0";
            if ((board.WhiteBoard & (1UL << position)) != 0) return WhitePiece + "0";
            return " ";
        }
    }
}
