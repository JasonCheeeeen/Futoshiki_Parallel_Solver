#include "puzzlegame.h"
#include "puzzleio.h"

#include <iostream>
#include <string>
#include <map>

using namespace std;

typedef bool (* check_value)(int, int *, int ***, vector<relation_operators>);
map<string, check_value> value_check = {
    {"RowsCols", check_rows_and_cols},
    {"RelationalOperator", check_relational_operators}
};

void find_empty_cell(int **location, int ***puzzle){
    
    *location = (int *)malloc(sizeof(int) * 2);
    (*location)[0] = -1;
    (*location)[1] = -1;
    for (int i = 0; i < puzzle_size; ++i){
        for (int j = 0; j < puzzle_size; ++j){
            if ((*puzzle)[i][j] == 0){
                (*location)[0] = i;
                (*location)[1] = j;
                return;
            }
        }
    }
    return;
}

bool check_rows_and_cols(int value, int *location, int ***puzzle, vector<relation_operators> puzzle_relation_operators){
    
    /* check row */
    for (int col = 0; col < puzzle_size; ++col){
        if ((*puzzle)[location[0]][col] == value){
            return false;
        }
    }

    /* check column */
    for (int row = 0; row < puzzle_size; ++row){
        if ((*puzzle)[row][location[1]] == value){
            return false;
        }
    }

    return true;
}

bool check_relational_operators(int value, int *location, int ***puzzle, vector<relation_operators> puzzle_relation_operators){
    for (auto ro : puzzle_relation_operators){
        if (ro.start.first == location[0] && ro.start.second == location[1]){

            /* the cell which have not insert value */
            if ((*puzzle)[ro.end.first][ro.end.second] == 0){
                continue;
            }

            if (ro.type == "larger"){
                if (value < (*puzzle)[ro.end.first][ro.end.second]){
                    return false;
                }
            }
            else{
                if (value > (*puzzle)[ro.end.first][ro.end.second]){
                    return false;
                }
            }
        }
        if (ro.end.first == location[0] && ro.end.second == location[1]){

            /* the cell which have not insert value */
            if ((*puzzle)[ro.start.first][ro.start.second] == 0){
                continue;
            }

            if (ro.type == "larger"){
                if (value > (*puzzle)[ro.start.first][ro.start.second]){
                    return false;
                }
            }
            else{
                if (value < (*puzzle)[ro.start.first][ro.start.second]){
                    return false;
                }
            }
        }
    }

    return true;
}

bool check = false;
void play_game_openmp(int ***puzzle, vector<relation_operators> puzzle_relation_operators){
    
    // puzzleout("filename", puzzle, puzzle_relation_operators);
    // cout << endl;

    /* find the empty cell */
    int *location;
    find_empty_cell(&location, puzzle);

    /* check finished or not */
    if (location[0] == -1 && location[1] == -1){
        check = true;
        return;
    }

    /* start to try all propable values */
    for (int value = 1; value <= puzzle_size; ++value){

        if (value_check["RowsCols"](value, location, puzzle, puzzle_relation_operators) == true &&
            value_check["RelationalOperator"](value, location, puzzle, puzzle_relation_operators) == true){
            
            /* push value */
            (*puzzle)[location[0]][location[1]] = value;

            /* recursion - DFS */
            int **undo_puzzle = (int **)malloc(sizeof(int *) * puzzle_size);
            for(int i = 0; i < puzzle_size; ++i) {
                undo_puzzle[i] = (int *)malloc(sizeof(int) * puzzle_size);
            }
            for(int i = 0; i < puzzle_size; i++) {
                for(int j = 0; j < puzzle_size; j++) {
                    undo_puzzle[i][j] = (*puzzle)[i][j];
                }
            }
            
            #pragma omp task
            play_game_openmp(&undo_puzzle, puzzle_relation_operators);
            if(check == true) {
                for(int i = 0; i < puzzle_size; i++) {
                    for(int j = 0; j < puzzle_size; j++) {
                        (*puzzle)[i][j] = undo_puzzle[i][j];
                    }
                }
                return;
            }
            else{
                for(int i = 0; i < puzzle_size; ++i) {
                    free(undo_puzzle[i]);
                }
                free(undo_puzzle);
            }

            /* this value cannot satisfy later operations, pop value */
            (*puzzle)[location[0]][location[1]] = 0;
        }
    }

    free(location);

    return;
}

bool play_game_sequential(int ***puzzle, vector<relation_operators> puzzle_relation_operators){
    
    // puzzleout("filename", puzzle, puzzle_relation_operators);
    // cout << endl;

    /* find the empty cell */
    int *location;
    find_empty_cell(&location, puzzle);

    /* check finished or not */
    if (location[0] == -1 && location[1] == -1){
        return true;
    }

    /* start to try all propable values */
    for (int value = 1; value <= puzzle_size; ++value){

        if (value_check["RowsCols"](value, location, puzzle, puzzle_relation_operators) == true &&
            value_check["RelationalOperator"](value, location, puzzle, puzzle_relation_operators) == true){
            
            /* push value */
            (*puzzle)[location[0]][location[1]] = value;

            /* recursion - DFS */
            if (play_game_sequential(puzzle, puzzle_relation_operators) == true){
                return true;
            }

            /* this value cannot satisfy later operations, pop value */
            (*puzzle)[location[0]][location[1]] = 0;
        }
    }

    free(location);

    return 0;
}