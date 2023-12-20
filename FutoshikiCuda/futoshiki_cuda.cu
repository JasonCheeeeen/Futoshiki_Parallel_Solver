#include "puzzlecuda.cuh"

#include <cuda.h>

#include <iostream>
#include <string>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

using namespace std;

typedef void (* io_puzzle_cuda) (FILE *, char **, int **);
map<string, io_puzzle_cuda> puzzle_io_cuda = {
    {"in", puzzlein_cuda},
    {"out", puzzleout_cuda}
};

int main(int argc, char *argv[]) {

    /* parse txt file name */
    if(argc == 1){
        cerr << "Please input the futoshiki game setting !!\n";
        cerr << "usage : ./f... {name of dataset stored in dataset}\n";
        return 1;
    }

    cudaFree(0);

    /* check cuda device */
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Error code: %d\n", cudaStatus);
        return 1;
    }

    /* process file name */
    char *filename;
    char filename_prefix[] = "../dataset/";
    filename = (char *)malloc(sizeof(char) * (strlen(filename_prefix) + strlen(argv[1]) + 1));
    strcpy(filename, filename_prefix);
    strcat(filename, argv[1]);

    FILE *in = fopen(filename, "r");
    if (in == NULL) {
        fprintf(stderr, "Can't open input file %s!\n",argv[1]);
        exit(1);
    }

    /* read input puzzle and relation operation */
    char *puzzle = NULL;
    int *relation_operation = NULL;

    puzzle_io_cuda["in"](in, &puzzle, &relation_operation);

    /* output the puzzle originally */
    puzzle_io_cuda["out"](in, &puzzle, &relation_operation);

    /* start timer */
    auto start_time = chrono::high_resolution_clock::now();

    puzzle_cuda_solver(puzzle, relation_operation);
    
    /* end timer */
    auto end_time = std::chrono::high_resolution_clock::now();

    /* compute execution time */
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    cout << "Time taken : " << (double) duration.count() / 1e6 << " seconds" << endl;

    /* output the puzzle result */
    puzzle_io_cuda["out"](in, &puzzle, &relation_operation);

    /* free memory */
    free(puzzle);
    free(relation_operation);

    /* close file */
    fclose(in);

}