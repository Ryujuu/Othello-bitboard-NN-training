﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Othello_bitboard_NN_training
{
    public class BenchmarkBot // Change out this benchmark bot to the best bot from the last training session or something
    {
        private int DepthLimit;
        public int Rating { get; set; }

        public BenchmarkBot()
        {
            DepthLimit = 6;
            Rating = 1000;
        }

        public Move GetMove(Board board, Player currentPlayer)
        {
            Move bestMove;

            bestMove = Search(board.Clone(), DepthLimit, currentPlayer, double.MinValue, double.MaxValue);


            return bestMove;
        }

        private Move Search(Board board, int depth, Player currentPlayer, double alpha, double beta)
        {
            if (depth == 0 || board.IsGameOver())
            {
                double evaluation = new Evaluation().EvaluateBoard(board);
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

        public int PlayGame(Bot bot1, bool isBot1Black)
        {
            Board board = new();

            // Determine who should start first
            Player currentPlayer = new(isBot1Black, isBot1Black);
            Player opponentPlayer = currentPlayer.GetOpponent();
            bool isBlack = true;

            while (!board.IsGameOver())
            {
                Move move;

                if (currentPlayer.IsBlack == isBlack && board.GenerateValidMoves(currentPlayer.IsBlack) != 0)
                {
                    move = bot1.GetMove(board, currentPlayer);
                    board.MakeMove(move.Position, currentPlayer.IsBlack);
                }
                else if (board.GenerateValidMoves(opponentPlayer.IsBlack) != 0)
                {
                    move = GetMove(board, opponentPlayer);
                    board.MakeMove(move.Position, opponentPlayer.IsBlack);
                }

                // Switch players
                isBlack = !isBlack;
            }

            int blackDiscs = board.CountBits(board.BlackBoard);
            int whiteDiscs = board.CountBits(board.WhiteBoard);

            // Determine the winner
            if (blackDiscs > whiteDiscs)
                return isBot1Black ? -1 : 1; // Bot 2 wins if Bot 1 plays black, otherwise Bot 1 wins
            else if (whiteDiscs > blackDiscs)
                return isBot1Black ? 1 : -1; // Bot 1 wins if Bot 1 plays black, otherwise Bot 2 wins
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
    public class Evaluation
    {
        // Weights for different evaluation components
        public double MobilityWeight { get; set; } = 7.0;
        public double CornerWeight { get; set; } = 30.0;
        public double AdjacentCornerWeight { get; set; } = -20.0;
        public double DiscWeight { get; set; } = 1.0;
        public double WinScore { get; set; } = 1000.0;

        private int[] corners = { 0, 7, 56, 63 };
        private int[][] adjacentToCorners = {
        new int[] { 1, 8, 9 }, new int[] { 6, 14, 15 },
        new int[] { 48, 49, 57 }, new int[] { 54, 55, 62 }
    };

        public double EvaluateBoard(Board board)
        {
            // Variables to accumulate scores
            double mobilityScore = 0;
            double cornerScore = 0;
            double adjacentCornerScore = 0;
            double discScore = 0;
            double terminalScore = 0;

            // Determine current player and opponent boards
            ulong blackBoard = board.BlackBoard;
            ulong whiteBoard = board.WhiteBoard;

            int blackDiscs = CountBits(blackBoard);
            int whiteDiscs = CountBits(whiteBoard);

            // Check for terminal state (win/loss)
            if (board.IsGameOver())
            {
                if (blackDiscs > whiteDiscs)
                {
                    terminalScore = WinScore; // Black wins (positive score)
                }
                else if (whiteDiscs > blackDiscs)
                {
                    terminalScore = -WinScore; // White wins (negative score)
                }
                else
                {
                    terminalScore = 0; // Draw
                }
                return terminalScore;
            }

            // Mobility: Number of valid moves available
            ulong blackValidMoves = board.GenerateValidMoves(true);
            ulong whiteValidMoves = board.GenerateValidMoves(false);
            mobilityScore = (CountBits(blackValidMoves) - CountBits(whiteValidMoves)) * MobilityWeight;

            // Corners
            foreach (int corner in corners)
            {
                if ((blackBoard & (1UL << corner)) != 0) cornerScore += CornerWeight;
                if ((whiteBoard & (1UL << corner)) != 0) cornerScore -= CornerWeight;
            }

            // Adjacent to corners
            for (int i = 0; i < adjacentToCorners.Length; i++)
            {
                foreach (int adj in adjacentToCorners[i])
                {
                    if ((blackBoard & (1UL << adj)) != 0) adjacentCornerScore -= AdjacentCornerWeight;
                    if ((whiteBoard & (1UL << adj)) != 0) adjacentCornerScore += AdjacentCornerWeight;
                }
            }

            // Discs
            discScore = (blackDiscs - whiteDiscs) * DiscWeight;

            // Calculate total score
            double totalScore = mobilityScore + cornerScore + adjacentCornerScore + discScore + terminalScore;
            double noise = (new Random().NextDouble() - 0.5) * totalScore * 0.01;
            return totalScore + noise;
        }

        private int CountBits(ulong bitboard)
        {
            int count = 0;
            while (bitboard != 0)
            {
                bitboard &= (bitboard - 1); // Clear the least significant bit set
                count++;
            }
            return count;
        }
    }
}
