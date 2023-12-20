#include <puzzleio.h>
#include <puzzlegame.h>

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <map>

using namespace std;

/* puzzle io function pointer collection */
typedef void (* io_puzzle)(string, int ***, vector<relation_operators> &);
map<string, io_puzzle> puzzle_io = {
    {"in", puzzlein},
    {"out", puzzleout}
};

int main(int argc, char *argv[]){
    
    /* parse txt file name */
    if(argc == 1){
        cerr << "Please input the futoshiki game setting !!\n";
        cerr << "usage : ./f... {name of dataset stored in dataset}\n";
        return 1;
    }
    string filename = argv[1];

    /* parse txt file to get values and relation_operators */
    int **puzzle = NULL;
    vector<relation_operators> puzzle_relation_operators;

    /* puzzle in process */
    puzzle_io["in"](filename, &puzzle, puzzle_relation_operators);

    /* puzzle out process - original */
    puzzle_io["out"](filename, &puzzle, puzzle_relation_operators);

    /* start timer */
    auto start_time = chrono::high_resolution_clock::now();

    /* play the game */
    #pragma omp single
    play_game_openmp(&puzzle, puzzle_relation_operators);

    /* end timer */
    auto end_time = std::chrono::high_resolution_clock::now();

    /* compute execution time and output */
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    cout << "Time taken : " << (double) duration.count() / 1e6 << " seconds" << endl;

    /* puzzle out process - solved */
    puzzle_io["out"](filename, &puzzle, puzzle_relation_operators);

}