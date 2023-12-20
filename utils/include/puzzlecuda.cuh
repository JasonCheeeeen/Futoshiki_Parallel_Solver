#ifndef PUZZLECUDA_CUH
#define PUZZLECUDA_CUH

#include <stdio.h>

extern int puzzle_size, relation_operation_size;

/* cpu side function */
void puzzlein_cuda(FILE *fp, char **puzzle, int **relation_operation);
void puzzleout_cuda(FILE *fp, char **puzzle, int **relation_operation);
void puzzle_cuda_solver(char *arr, int *relation_operation);

/* gpu side function */
__global__ void controller(char* arr_dev, int *block_stat, int *relat_op, int nBlocks, int nThreads, int puzzle_size, int relation_operation_size);
__global__ void puzzle_cuda_bfs(char *memory, int *stats, int *relat_op, int puzzle_size, int relation_operation_size);

#endif // PUZZLECUDA_CUH