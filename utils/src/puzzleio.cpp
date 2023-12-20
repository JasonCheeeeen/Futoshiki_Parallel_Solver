#include "puzzleio.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

/* store value into extern value */
int puzzle_size;

void puzzlein(string filename, int ***puzzle, vector<relation_operators> &puzzle_relation_operators){

    /* ifstream the file */
    string filepath = "../dataset/" + filename;
    ifstream ifs(filepath);
    if (!ifs.is_open()){
        cerr << "unable to open file !!\n";
        exit(1);
    }

    string line;

    /* get puzzle size */
    getline(ifs, line);
    puzzle_size = stoi(line);

    /* create puzzle */
    *puzzle = (int **)calloc(puzzle_size, sizeof(int *));
    for (int i = 0; i < puzzle_size; ++i){
        (*puzzle)[i] = (int *)calloc(puzzle_size, sizeof(int));
    }

    /* nonzero value */
    getline(ifs, line);
    int num_nonzero = stoi(line);

    /* insert value into puzzle */
    for (int i = 0 ; i < num_nonzero ; ++i){
        getline(ifs, line);
        string s;
        stringstream ss(line);
        vector<int> vi;
        while(ss >> s){
            vi.push_back(stoi(s));
        }
        (*puzzle)[vi[0]][vi[1]] = vi[2];
    }

    /* relation_operators */
    getline(ifs, line);
    int num_relation_operators = stoi(line);

    /* insert relation_operators into relation_operators' struct */
    for (int i = 0; i < num_relation_operators; ++i){
        getline(ifs, line);
        string s;
        stringstream ss(line);
        vector<int> vi;
        while(ss >> s){
            vi.push_back(stoi(s));
        }
        relation_operators single;
        single.start.first = vi[0];
        single.start.second = vi[1];
        single.end.first = vi[2];
        single.end.second = vi[3];
        single.type = (vi[4] == 1) ? "larger" : "lower";
        puzzle_relation_operators.push_back(single);
    }
}

void puzzleout(string filename, int ***puzzle, vector<relation_operators> &puzzle_relation_operators){

    cout << endl;

    /* process output each of line */
    for (int i = 0; i < puzzle_size; ++i){
        string output = "";
        for (int j = 0; j < puzzle_size; ++j){
            if ((*puzzle)[i][j] == 0) {
                output = output + ". ";
            }
            else {
                output = output + to_string((*puzzle)[i][j]) + " ";
            }
            bool bool_ro = false;
            for (auto ro : puzzle_relation_operators){
                if (ro.start.first == i && ro.start.second == j && ro.start.first == ro.end.first){
                    output = output + (ro.type == "larger" ? "> " : "< ");
                    bool_ro = true;
                    break;
                }
            }
            output = output + (bool_ro == 1 ? "" : "  ");
        }
        cout << output << endl;
        output = "";
        for (int j = 0; j < puzzle_size; ++j){
            bool bool_ro = false;
            for (auto ro : puzzle_relation_operators){
                if (ro.start.first == i && ro.end.second == j && ro.start.second == ro.end.second){
                    output = output + (ro.type == "larger" ? "v" : "^");
                    bool_ro = true;
                    break;
                }
            }
            if (bool_ro == false){
                output += " ";
            }
            output += "   ";
        }
        cout << output << endl;
    }

}