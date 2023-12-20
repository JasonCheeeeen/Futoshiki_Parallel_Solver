#include "puzzlecuda.cuh"

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

int puzzle_size, relation_operation_size;

__device__ int lock = 0;

__global__ void puzzle_cuda_bfs(char *memory, int *stats, int *relat_op, int puzzle_size, int relation_operation_size) {
    uint i, j, mat_i, mat_j, k, temp, current_poss;
    char *block_memory = memory + ((puzzle_size * puzzle_size) * blockIdx.x);

    /* cuda share memory */
    extern __shared__ uint shared_mem[];

    uint *row_used_numbers = shared_mem;
    uint *col_used_numbers = row_used_numbers + puzzle_size;

    if (blockIdx.x == 0 && threadIdx.x < puzzle_size){
        row_used_numbers[threadIdx.x] = 0;
        col_used_numbers[threadIdx.x] = 0;
    }

    __syncthreads();

    __shared__ char progress_flag;
    __shared__ char done_flag;
    __shared__ char error_flag;
    __shared__ int min_forks;
    __shared__ int scheduling_thread;

    if (stats[blockIdx.x] == 1) {
        if (threadIdx.x == 0) {
            error_flag = 0;
            done_flag = 0;
            progress_flag = 1;
        }

        __syncthreads();

        while (progress_flag && !done_flag && !error_flag) {
            // printf("%d %d / %d\n", threadIdx.x, blockIdx.x, gridDim.x);
            __syncthreads();

            if (threadIdx.x < puzzle_size) {
                
                /* prepare the value can insert in row and col which apply with constraints */
                row_used_numbers[threadIdx.x] = 0;
                col_used_numbers[threadIdx.x] = 0;

                for (i = 0; i < puzzle_size; ++i){

                    /* check rows */
                    temp = block_memory[threadIdx.x * puzzle_size + i];
                    if (temp) {
                        if ((row_used_numbers[threadIdx.x] >> (temp - 1)) & 1){
                            /* same number is the same row */
                            error_flag = 10 + i;
                        }
                        /* set bit to 1 */
                        row_used_numbers[threadIdx.x] |= 1 << (temp - 1);
                    }

                    /* check cols */
                    temp = block_memory[i * puzzle_size + threadIdx.x];
                    if (temp) {
                        if ((col_used_numbers[threadIdx.x] >> (temp - 1)) & 1){
                            /* same number is the same col */
                            error_flag = 20 + i;
                        }
                        /* set bit to 1 */
                        col_used_numbers[threadIdx.x] |= (1 << (temp - 1));
                    }
                }
            }
            
            __syncthreads();

            if (error_flag == 0){
                if (threadIdx.x==0) {
                    progress_flag = 0;
                    done_flag = 1;
                }

                __syncthreads();

                if (threadIdx.x < (puzzle_size * puzzle_size)) {
                    current_poss = 0;
                    mat_i = threadIdx.x / puzzle_size;
                    mat_j = threadIdx.x % puzzle_size;
                    if (block_memory[threadIdx.x] == 0) {
                        done_flag = 0;
                        current_poss = (row_used_numbers[mat_i] | col_used_numbers[mat_j]);

                        /* check constraints */
                        if (relat_op[0] != 0) {
                            for (i = 0; i < relation_operation_size; ++i) {
                                if (relat_op[4 * i + 1] == mat_i && relat_op[4 * i + 2] == mat_j) {
                                    int compare_value = block_memory[relat_op[4 * i + 3] * puzzle_size + relat_op[4 * i + 4]];
                                    if (compare_value != 0) {
                                        for (int val = 1; val <= compare_value; ++val) {
                                            current_poss |= (1 << (val-1));
                                        }
                                    }
                                }
                                if (relat_op[4 * i + 3] == mat_i && relat_op[4 * i + 4] == mat_j) {
                                    int compare_value = block_memory[relat_op[4 * i + 1] * puzzle_size + relat_op[4 * i + 2]];
                                    if (compare_value != 0) {
                                        for (int val = compare_value; val <= puzzle_size; ++val) {
                                            current_poss |= (1 << (val-1));
                                        }
                                    }
                                }
                            }
                        }

                        temp = 0;
                        for (i = 0; i < puzzle_size; ++i){
                            if ((current_poss & (1 << i)) == 0){
                                if (temp){
                                    temp = puzzle_size + 1;
                                    break;
                                }
                                else{
                                    temp = i + 1;
                                }
                            }
                        }
                        if (temp == 0){
                            #ifdef DEBUG
                                printf("Block:%d,i=%d,j=%d, cannot be filled. Invalidating\n", blockIdx.x, mat_i+1, mat_j+1);
                            #endif
                            error_flag = 1;
                            progress_flag = 1;
                        }
                        else if (temp <= puzzle_size){
                            #ifdef DEBUG
                                if (blockIdx.x == 0)  printf("i=%d, j=%d, val=%d\n", threadIdx.x / puzzle_size + 1, (threadIdx.x % puzzle_size) + 1, temp);
                            #endif
                            block_memory[threadIdx.x] = temp;
                            progress_flag = 1;
                        }
                    }
                }
            }

            __syncthreads();

        }

        __syncthreads();

        if (done_flag) {
            if (threadIdx.x == 0 && stats[gridDim.x] != 2) {

                /* critical section */
                while (atomicExch(&lock, 1) != 0) {

                }

                memcpy(memory + gridDim.x * (puzzle_size * puzzle_size), block_memory, puzzle_size * puzzle_size);
                stats[gridDim.x] = 2;
                
                /* since each thread will modify the value, it need to recheck the constraints */
                if (relat_op[0] != 0) {
                    for (i = 0; i < relation_operation_size; ++i) {
                        if (block_memory[relat_op[4 * i + 1] * puzzle_size + relat_op[4 * i + 2]] <= block_memory[relat_op[4 * i + 3] * puzzle_size + relat_op[4 * i + 4]]) {
                            stats[gridDim.x] = 0;
                            break;
                        }
                    }
                }

                atomicExch(&lock, 0);
            }
        }
        else if (error_flag != 0){
            #ifdef DEBUG
                if (threadIdx.x == 0) {
                    printf("There is an error:%d : with this block %d \n",error_flag,blockIdx.x);
                }
            #endif
            if (threadIdx.x == 0) {
                stats[blockIdx.x] = 0;
            }
        }
        else if (progress_flag == 0) {

            /* implement schedule */
            if (threadIdx.x == 0){
                min_forks = puzzle_size;
                scheduling_thread = blockDim.x;
            }

            __syncthreads();

            /* find which cell has minimum possible value */
            temp = 0;
            if(current_poss != 0){
                for (i = 0; i < puzzle_size; ++i){
                    if ((current_poss & (1<<i)) == 0){
                        temp++;
                    }
                }
                atomicMin(&min_forks, temp);
            }

            __syncthreads();

            /* find the minimum thread id with minimum possible value in cell */
            if (temp == min_forks){
                atomicMin(&scheduling_thread, threadIdx.x);
            }

            __syncthreads();


            if (scheduling_thread == threadIdx.x){

                /* find suitable block to schedule for each propable value */
                k = 1;
                j = 0;
                for (i = 0; i < puzzle_size; ++i){
                    if ((current_poss & (1 << i)) == 0){
                        if (k == 1) {
                            
                            /* insert probable value into first block to test continually */
                            block_memory[threadIdx.x] = i + 1;
                        }
                        else{
                            
                            /* allocate probable value to each block - bfs */
                            for (; j < gridDim.x; ++j) {
                                atomicCAS(stats + j, 0, gridDim.x * blockIdx.x + threadIdx.x + 2);
                                if (stats[j] == (gridDim.x * blockIdx.x + threadIdx.x + 2)){
                                    memcpy(memory + j * (puzzle_size * puzzle_size), block_memory, (puzzle_size * puzzle_size));
                                    memory[j * (puzzle_size * puzzle_size) + threadIdx.x] = i + 1;
                                    stats[j] = 1;
                                    break;
                                }
                            }
                        }
                        ++k;
                    }
                }
            }

            __syncthreads();

        __syncthreads();

        }
    }
}

__global__ void controller(char* arr_dev, int *block_stat, int *relat_op, int nBlocks, int nThreads, int puzzle_size, int relation_operation_size) {
    while (block_stat[nBlocks] != 2){
        puzzle_cuda_bfs<<<nBlocks,nThreads>>>(arr_dev, block_stat, relat_op, puzzle_size, relation_operation_size);
  }
}

void puzzle_cuda_solver(char *arr, int *relation_operation) {

    char *arr_dev;
    int *block_stat;
    int *relat_op;

    /* warp size = 32 */
    int nThreads = (puzzle_size * puzzle_size);

    /* max available concurrent blocks/searches running */
    int nBlocks = 20000; 

    /* total memory size */
    int memSize = (puzzle_size * puzzle_size) * (nBlocks + 1);
    
    /* copy memory from host to cuda device */

    /* stats */
    cudaMalloc((void**) &block_stat, (nBlocks + 1) * sizeof(int));
    cudaMemset(block_stat, 0, (nBlocks+1)*sizeof(int));
    cudaMemset(block_stat, 1, 1);
    if (!block_stat) {
        fprintf(stderr, " Cannot allocate block_stat array of size %d on the device\n", (int)(nBlocks+1) * (int)sizeof(int));
        exit(1);
    }

    /* puzzle */
    cudaMalloc((void**) &arr_dev, memSize);
    cudaMemcpy(arr_dev, arr, puzzle_size * puzzle_size, cudaMemcpyHostToDevice);
    if (!arr_dev) {
        fprintf(stderr, " Cannot allocate arr_dev of size %d on the device\n", memSize);
        exit(1);
    }

    /* relation operation */
    cudaMalloc((void**) &relat_op, sizeof(int) * 4 * relation_operation_size + 1);
    cudaMemcpy(relat_op, relation_operation, sizeof(int) * 4 * relation_operation_size + 1, cudaMemcpyHostToDevice);
    if (!relat_op){
        fprintf(stderr, " Cannot allocate relation operation of size %d on the device\n", 4 * relation_operation_size);
        exit(1);
    }


    printf("Block=%d, threads=%d starting\n", nBlocks, nThreads);
    controller<<<1,1>>>(arr_dev, block_stat, relat_op, nBlocks, nThreads, puzzle_size, relation_operation_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));

    cudaMemcpy(arr, arr_dev + (puzzle_size * puzzle_size) * nBlocks, (puzzle_size * puzzle_size), cudaMemcpyDeviceToHost);
    cudaFree(arr_dev);
    cudaFree(block_stat);
}

void puzzlein_cuda(FILE *fp, char **puzzle, int **relation_operation){
    size_t size;
    
    /* get puzzle size */
    fscanf(fp, "%zu\n", &size);
    puzzle_size = (int)size;

    /* allocate puzzle memory */
    *puzzle = (char *)malloc(sizeof(char) * (puzzle_size * puzzle_size));

    /* get num of non zero place */
    fscanf(fp, "%zu\n", &size);

    /* insert non zero value */
    for (int i = 0; i < size; ++i) {
        int row, col;
        char val;
        fscanf(fp, "%d %d %c\n", &row, &col, &val);
        (*puzzle)[row * puzzle_size + col] = val - 48;
    }

    /* get num of relation operation */
    fscanf(fp, "%zu\n", &size);
    relation_operation_size = (int)size;

    /* allocate relation operation memory */
    *relation_operation = (int *)malloc(sizeof(int) * (4 * size + 1));

    /* relation operation flag */
    if (size == 0){
        (*relation_operation)[0] = 0;
    }
    else {
        (*relation_operation)[0] = 1;
    }

    /* insert relation operation location */
    for (int i = 0; i < size; ++i) {
        int row1, row2, col1, col2, op;
        fscanf(fp, "%d %d %d %d %d\n", &row1, &col1, &row2, &col2, &op);
        if (op == 1) {
            (*relation_operation)[i * 4 + 1] = row1;
            (*relation_operation)[i * 4 + 2] = col1;
            (*relation_operation)[i * 4 + 3] = row2;
            (*relation_operation)[i * 4 + 4] = col2;
        }
        else {
            (*relation_operation)[i * 4 + 1] = row2;
            (*relation_operation)[i * 4 + 2] = col2;
            (*relation_operation)[i * 4 + 3] = row1;
            (*relation_operation)[i * 4 + 4] = col1;
        }
    }
}

void puzzleout_cuda(FILE *fp, char **puzzle, int **relation_operation){
    printf("\n");
    for (int i = 0; i < puzzle_size; ++i) {
        char output[puzzle_size + 3 * (puzzle_size-1)];
        memset(output, ' ', sizeof(output));
        for (int j = 0; j < puzzle_size; ++j) {
            output[4 * j] = (*puzzle)[i * puzzle_size + j];
        }
        for (int k = 0; k < relation_operation_size; ++k) {
            if (i == (*relation_operation)[4 * k + 1] && (*relation_operation)[4 * k + 1] == (*relation_operation)[4 * k + 3]) {
                if ((*relation_operation)[4 * k + 2] < (*relation_operation)[4 * k + 4]) {
                    output[4 * (*relation_operation)[4 * k + 2] + 2] = '>';
                }
                else {
                    output[4 * (*relation_operation)[4 * k + 4] + 2] = '<';
                }
            }
        }
        for (int l = 0; l < sizeof(output); ++l) {
            if (output[l] == 0) {
                printf(".");
            }
            else if (output[l] != ' ' && output[l] != '>' && output[l] != '<') {
                printf("%d", output[l]);
            }
            else {
                printf("%c", output[l]);
            }
        }

        printf("\n");

        if (i != puzzle_size - 1) {
            memset(output, ' ', sizeof(output));
            for (int k = 0; k < relation_operation_size; ++k) {
                if (((i == (*relation_operation)[4 * k + 1] && (i + 1) == (*relation_operation)[4 * k + 3]) || (i == (*relation_operation)[4 * k + 3] && (i + 1) == (*relation_operation)[4 * k + 1])) && (*relation_operation)[4 * k + 2] == (*relation_operation)[4 * k + 4]) {
                    if ((*relation_operation)[4 * k + 1] < (*relation_operation)[4 * k + 3]) {
                        output[4 * (*relation_operation)[4 * k + 2]] = 'v';
                    }
                    else {
                        output[4 * (*relation_operation)[4 * k + 2]] = '^';
                    }
                }
            }
            for (int l = 0; l < sizeof(output); ++l) {
                if (output[l] == 0) {
                    printf(".");
                }
                else if (output[l] != ' ' && output[l] != '^' && output[l] != 'v') {
                    printf("%d", output[l]);
                }
                else {
                    printf("%c", output[l]);
                }
            }
            printf("\n");
        }
    }
    printf("\n\n");
}