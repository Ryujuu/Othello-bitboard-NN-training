using Othello_bitboard_NN_training;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Othello_bitboard_NN_training
{
    public class Board
    {
        // Properties for black and white bitboards
        public ulong BlackBoard { get; set; }
        public ulong WhiteBoard { get; set; }

        // Directions for move generation
        private static readonly int[] Directions = { -1, -8, 1, 8, -9, -7, 7, 9 };

        public Board()
        {
            InitializeBoard();
        }
        // Cloning method to create a copy of the board
        public Board Clone()
        {
            return new Board
            {
                BlackBoard = this.BlackBoard,
                WhiteBoard = this.WhiteBoard
            };
        }

        // Initialize the board with a starting position
        public void InitializeBoard()
        {
            // Standard starting position
            BlackBoard = 0x0000000810000000;
            WhiteBoard = 0x0000001008000000;
        }

        public void MakeMove(int move, bool isBlack)
        {
            ulong playerBoard = isBlack ? BlackBoard : WhiteBoard;
            ulong opponentBoard = isBlack ? WhiteBoard : BlackBoard;
            ulong moveMask = 1UL << move;
            playerBoard |= moveMask;
            ulong flippedPieces = 0;

            foreach (int direction in Directions)
            {
                ulong potentialFlips = 0;
                ulong shifted = Shift(moveMask, direction);

                while ((shifted & opponentBoard) != 0)
                {
                    potentialFlips |= shifted;
                    shifted = Shift(shifted, direction);
                }

                if ((shifted & playerBoard) != 0)
                {
                    flippedPieces |= potentialFlips;
                }
            }

            // Update boards
            playerBoard |= flippedPieces;
            opponentBoard &= ~flippedPieces;

            // Update the board object with the new state
            if (isBlack)
            {
                BlackBoard = playerBoard;
                WhiteBoard = opponentBoard;
            }
            else
            {
                WhiteBoard = playerBoard;
                BlackBoard = opponentBoard;
            }
        }

        public ulong GenerateValidMoves(bool isBlack)
        {
            ulong playerBoard = isBlack ? BlackBoard : WhiteBoard;
            ulong opponentBoard = isBlack ? WhiteBoard : BlackBoard;
            ulong emptySpaces = ~(playerBoard | opponentBoard);
            ulong validMoves = 0;

            foreach (int direction in Directions)
            {
                ulong potentialFlips = 0;
                ulong opponentPiecesInDirection = Shift(playerBoard, direction) & opponentBoard;

                while (opponentPiecesInDirection != 0)
                {
                    potentialFlips |= opponentPiecesInDirection;
                    opponentPiecesInDirection = Shift(opponentPiecesInDirection, direction) & opponentBoard;
                }

                ulong potentialMoves = Shift(potentialFlips, direction) & emptySpaces;
                validMoves |= potentialMoves;
            }

            return validMoves;
        }

        public bool IsValidMove(int move, bool isBlack)
        {
            ulong validMoves = GenerateValidMoves(isBlack);
            return (validMoves & (1UL << move)) != 0;
        }

        public bool IsGameOver()
        {
            // Check if there are any valid moves for either player
            bool blackHasMoves = HasValidMoves(true);
            bool whiteHasMoves = HasValidMoves(false);

            // If neither player has valid moves, the game is over
            if (!blackHasMoves && !whiteHasMoves)
                return true;

            // Alternatively, if the board is full, the game is over
            ulong allPieces = BlackBoard | WhiteBoard;
            if (CountBits(allPieces) == 64)
                return true;

            return false;
        }

        public double EvaluateGameResult(bool isBlack)
        {
            int blackScore = CountBits(BlackBoard);
            int whiteScore = CountBits(WhiteBoard);

            // Determine the result from the perspective of the player
            if (isBlack)
            {
                if (blackScore > whiteScore) return 1.0; // Win
                if (blackScore < whiteScore) return -1.0; // Loss
            }
            else
            {
                if (whiteScore > blackScore) return 1.0; // Win
                if (whiteScore < blackScore) return -1.0; // Loss
            }

            return 0.0; // Draw
        }

        private bool HasValidMoves(bool isBlackTurn)
        {
            ulong validMoves = GenerateValidMoves(isBlackTurn);
            return validMoves != 0;
        }

        public int CountBits(ulong bitboard)
        {
            int count = 0;
            while (bitboard != 0)
            {
                bitboard &= (bitboard - 1); // Clear the least significant bit set
                count++;
            }
            return count;
        }

        private ulong Shift(ulong bitboard, int direction)
        {
            return direction switch
            {
                -1 => (bitboard >> 1) & 0x7F7F7F7F7F7F7F7F, // Left
                1 => (bitboard << 1) & 0xFEFEFEFEFEFEFEFE, // Right
                -8 => bitboard >> 8,                      // Up
                8 => bitboard << 8,                       // Down
                -9 => (bitboard >> 9) & 0x7F7F7F7F7F7F7F7F, // Up-left
                -7 => (bitboard >> 7) & 0xFEFEFEFEFEFEFEFE, // Up-right
                9 => (bitboard << 9) & 0xFEFEFEFEFEFEFEFE, // Down-right
                7 => (bitboard << 7) & 0x7F7F7F7F7F7F7F7F, // Down-left
                _ => 0,
            };
        }
    }
}
