using Othello_bitboard_NN_training;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Othello_bitboard_NN_training
{
    public class BoardFeatures
    {
        public double Mobility { get; set; }
        public double Corners { get; set; }
        public double Discs { get; set; }
        public double Turn { get; set; }
        public double TotalPieces { get; set; }
        public double WinCondition { get; set; }
        public double XSquares { get; set; }
        public double CSquares { get; set; }

        public double[] ToArray()
        {
            return new double[] { Mobility, Corners, Discs, Turn, TotalPieces, WinCondition, XSquares, CSquares };
        }

        public BoardFeatures GetBoardFeatures(Board board, bool isBlack)
        {
            double mobility = CalculateMobility(board, isBlack);
            double cornersScore = CalculateCorners(board, isBlack);
            double discs = CountDiscs(board, isBlack);
            double totalPieces = CountTotalPieces(board);
            double winCondition = CalculateWinCondition(board, isBlack);
            double xSquares = CalculateXSquares(board, isBlack);
            double cSquares = CalculateCSquares(board, isBlack);

            return new BoardFeatures
            {
                Mobility = mobility,
                Corners = cornersScore,
                Discs = discs,
                Turn = isBlack ? 1.0 : 0.0,
                TotalPieces = totalPieces,
                WinCondition = winCondition,
                XSquares = xSquares,
                CSquares = cSquares
            };
        }

        private double CalculateWinCondition(Board board, bool isBlack)
        {
            if (!board.IsGameOver())
            {
                return 0; // Not a terminal state
            }

            int blackDiscs = board.CountBits(board.BlackBoard);
            int whiteDiscs = board.CountBits(board.WhiteBoard);

            if (blackDiscs > whiteDiscs)
            {
                return isBlack ? 1 : -1; // Black wins if it was black's turn
            }
            else if (whiteDiscs > blackDiscs)
            {
                return isBlack ? -1 : 1; // White wins if it was black's turn
            }
            else
            {
                return 0; // Draw
            }
        }

        private double CalculateMobility(Board board, bool isBlack)
        {
            return CountBits(board.GenerateValidMoves(isBlack));
        }


        private double CalculateCorners(Board board, bool isBlack)
        {
            int blackCorners = 0, whiteCorners = 0;
            ulong[] corners = { 0x8000000000000000UL, 0x0100000000000000UL, 0x0000000000000080UL, 0x0000000000000001UL };

            foreach (ulong corner in corners)
            {
                if ((board.BlackBoard & corner) != 0) blackCorners++;
                if ((board.WhiteBoard & corner) != 0) whiteCorners++;
            }

            return isBlack ? blackCorners - whiteCorners : whiteCorners - blackCorners;
        }


        private double CountDiscs(Board board, bool isBlack)
        {
            return isBlack ? board.CountBits(board.BlackBoard) : board.CountBits(board.WhiteBoard);
        }


        private double CountTotalPieces(Board board)
        {
            return board.CountBits(board.BlackBoard) + board.CountBits(board.WhiteBoard);
        }


        private double CalculateXSquares(Board board, bool isBlack)
        {
            ulong xSquaresMask = 0x0042000000004200;

            int blackXSquares = CountBits(board.BlackBoard & xSquaresMask);
            int whiteXSquares = CountBits(board.WhiteBoard & xSquaresMask);

            return isBlack ? blackXSquares - whiteXSquares : whiteXSquares - blackXSquares;
        }

        private double CalculateCSquares(Board board, bool isBlack)
        {
            ulong cSquaresMask = 0x4281000000008142;

            int blackCSquares = CountBits(board.BlackBoard & cSquaresMask);
            int whiteCSquares = CountBits(board.WhiteBoard & cSquaresMask);

            return isBlack ? blackCSquares - whiteCSquares : whiteCSquares - blackCSquares;
        }

        public int CountBits(ulong board)
        {
            int count = 0;
            while (board != 0)
            {
                count++;
                board &= (board - 1); // Clear the least significant bit set
            }
            return count;
        }

    }
}
