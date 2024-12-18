#include <stdio.h>

#include <stdlib.h>

#include <time.h>

#include <string.h>  // Added for memcpy



#ifndef BOARD_DIM

#define BOARD_DIM 13

#endif



int neighbors[] = {-(BOARD_DIM+2) + 1, -(BOARD_DIM+2), -1, 1, (BOARD_DIM+2), (BOARD_DIM+2) - 1};



struct hex_game {

    int board[(BOARD_DIM+2)*(BOARD_DIM+2)*2];

    int open_positions[BOARD_DIM*BOARD_DIM];

    int number_of_open_positions;

    int moves[BOARD_DIM*BOARD_DIM][2];  // Store moves for each player, with player and position

    int connected[(BOARD_DIM+2)*(BOARD_DIM+2)*2];

};



void hg_init(struct hex_game *hg) {

    for (int i = 0; i < BOARD_DIM+2; ++i) {

        for (int j = 0; j < BOARD_DIM+2; ++j) {

            hg->board[(i*(BOARD_DIM + 2) + j) * 2] = 0;

            hg->board[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;



            if (i > 0 && i < BOARD_DIM + 1 && j > 0 && j < BOARD_DIM + 1) {

                hg->open_positions[(i-1)*BOARD_DIM + j - 1] = i*(BOARD_DIM + 2) + j;

            }



            if (i == 0) {

                hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 1;

            } else {

                hg->connected[(i*(BOARD_DIM + 2) + j) * 2] = 0;

            }



            if (j == 0) {

                hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 1;

            } else {

                hg->connected[(i*(BOARD_DIM + 2) + j) * 2 + 1] = 0;

            }

        }

    }

    hg->number_of_open_positions = BOARD_DIM*BOARD_DIM;

}



int hg_connect(struct hex_game *hg, int player, int position) {

    hg->connected[position*2 + player] = 1;



    if (player == 0 && position / (BOARD_DIM + 2) == BOARD_DIM) {

        return 1;

    }



    if (player == 1 && position % (BOARD_DIM + 2) == BOARD_DIM) {

        return 1;

    }



    for (int i = 0; i < 6; ++i) {

        int neighbor = position + neighbors[i];

        if (hg->board[neighbor*2 + player] && !hg->connected[neighbor*2 + player]) {

            if (hg_connect(hg, player, neighbor)) {

                return 1;

            }

        }

    }

    return 0;

}



int hg_winner(struct hex_game *hg, int player, int position) {

    for (int i = 0; i < 6; ++i) {

        int neighbor = position + neighbors[i];

        if (hg->connected[neighbor*2 + player]) {

            return hg_connect(hg, player, position);

        }

    }

    return 0;

}



int hg_place_piece_randomly(struct hex_game *hg, int player, int move_count) {

    int random_empty_position_index = rand() % hg->number_of_open_positions;



    int empty_position = hg->open_positions[random_empty_position_index];



    hg->board[empty_position * 2 + player] = 1;



    // Record the move (player and position)

    hg->moves[move_count][0] = player;

    hg->moves[move_count][1] = empty_position;



    // Replace the used position with the last open position

    hg->open_positions[random_empty_position_index] = hg->open_positions[hg->number_of_open_positions-1];



    hg->number_of_open_positions--;



    return empty_position;

}



int hg_full_board(struct hex_game *hg) {

    return hg->number_of_open_positions == 0;

}



// Function to save the board state

void save_board_state(FILE *file, int *board, int winner) {

    // Save the board state

    for (int i = 1; i < BOARD_DIM + 1; ++i) {

        for (int j = 1; j < BOARD_DIM + 1; ++j) {

            int position = (i * (BOARD_DIM + 2) + j);

            if (board[position * 2] == 1) {

                fprintf(file, "1,");  // Black piece

            } else if (board[position * 2 + 1] == 1) {

                fprintf(file, "-1,"); // White piece

            } else {

                fprintf(file, "0,");  // Empty

            }

        }

    }

    // Write the winner: 1 for Black, 0 for White

    int winner_value = (winner == 0) ? 1 : 0;

    fprintf(file, "%d\n", winner_value);

}



int main() {

    struct hex_game hg;



    srand(time(0));



    // Open two files: one for training and one for testing

    FILE *train_file = fopen("hex_train.csv", "w");

    FILE *test_file = fopen("hex_test.csv", "w");

    if (train_file == NULL || test_file == NULL) {

        printf("Error opening one of the files for writing.\n");

        return 1;

    }



    // Print the header with "winner" as the last column to both files

    for (int i = 0; i < BOARD_DIM; ++i) {

        for (int j = 0; j < BOARD_DIM; ++j) {

            fprintf(train_file, "cell%d_%d,", i, j);

            fprintf(test_file, "cell%d_%d,", i, j);

        }

    }

    fprintf(train_file, "winner\n");

    fprintf(test_file, "winner\n");



    int winner = -1;

    int games = 50000;  // Number of games to simulate



    // Simulate multiple games

    for (int game = 0; game < games; ++game) {

        hg_init(&hg);
        // random starting player
        int player = rand() % 2;



        int num_states = 0;



        // Simulate the game until a winner is found

        while (!hg_full_board(&hg)) {

            hg_place_piece_randomly(&hg, player, num_states);

            num_states++;



            if (hg_winner(&hg, player, hg.moves[num_states - 1][1])) {

                winner = player;



                // Save the current board state (terminal state) and the winner

                if (rand() % 100 < 67) {

                    save_board_state(train_file, hg.board, winner);

                } else {

                    save_board_state(test_file, hg.board, winner);

                }

                break;

            }



            player = 1 - player;

        }

    }



    fclose(train_file);

    fclose(test_file);

    return 0;

}