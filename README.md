# Futoshiki Parallel Solver
- It supports three kinds of way to solve the futoshiki game which include ***sequential***, ***openmp*** and ***cuda***.

## How to Run on your local machine ?
- clone the project
  ```
  $ git clone https://github.com/JasonCheeeeen/Futoshiki_Parallel_Solver.git
  ```
- Sequential
  ```
  $ cd Futoshiki_Parallel_Solver/FutoshikiSequential
  $ make clean && make
  $ ./futoshiki_sequential {name of data in dataset folder}
  ```
- OpenMP
  ```
  $ cd Futoshiki_Parallel_Solver/FutoshikiOpenmp
  $ make clean && make
  $ ./futoshiki_openmp {name of data in dataset folder}
  ```
- Sequential
  ```
  $ cd Futoshiki_Parallel_Solver/FutoshikiCuda
  $ make clean && make
  $ ./futoshiki_cuda {name of data in dataset folder}
  ```
