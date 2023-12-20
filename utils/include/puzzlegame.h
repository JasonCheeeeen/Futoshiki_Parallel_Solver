#ifndef PUZZLEGAME_H
#define PUZZLEGAME_H

#include "puzzleio.h"

#include <iostream>
#include <vector>

using namespace std;

/* game main function */
bool play_game_sequential(int ***puzzle, vector<relation_operators> puzzle_relation_operators);
void play_game_openmp(int ***puzzle, vector<relation_operators> puzzle_relation_operators);

/* check value in puzzle whether apply with the game's relation_operators */
bool check_rows_and_cols(int value, int *location, int ***puzzle, vector<relation_operators> puzzle_relation_operators);
bool check_relational_operators(int value, int *location, int ***puzzle, vector<relation_operators> puzzle_relation_operators);

/* find empty cell which need to fill the value */
void find_empty_cell(int **location, int ***puzzle);

#endif // PUZZLEGAME_H