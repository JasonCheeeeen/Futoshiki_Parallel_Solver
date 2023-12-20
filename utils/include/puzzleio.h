#ifndef PUZZLEIO_H
#define PUZZLEIO_H

#include <string>
#include <vector>
#include <map>

using namespace std;

extern int puzzle_size;

/* struct of constraints : > */
struct relation_operators{
    pair<int, int> start;
    pair<int, int> end;
    string type;
};
typedef struct relation_operators relation_operators;

/* functions of puzzles' input and output */
void puzzlein(string filename, int ***puzzle, vector<relation_operators> &puzzle_relation_operators);
void puzzleout(string filename, int ***puzzle, vector<relation_operators> &puzzle_relation_operators);

#endif //PUZZLEIO_H