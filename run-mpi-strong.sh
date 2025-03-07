#!/bin/bash
make

# Set constant problem size
n_particles=1000  # Adjust to desired size for strong scaling test

# Number of tasks per node for each scaling step
tasks_per_node_values=(1 2 4 8 16 32 64)

# Strong scaling test — fixed problem size, increasing processor counts
# First test case: 1 rank (special case, 1 node, 1 task per node)
echo "Running with -N 1 --ntasks-per-node=1, -n $n_particles (1 MPI rank)"
for i in {1..1}; do
    srun -N 1 --ntasks-per-node=1 ./mpi -n $n_particles -s 1
done

# Remaining test cases: 2 nodes, 2-64 tasks per node (covering 2, 4, 8, 16, 32, 64, 128 ranks)
for tasks_per_node in "${tasks_per_node_values[@]}"; do
    total_ranks=$((2 * tasks_per_node))
    echo "Running with -N 2 --ntasks-per-node=$tasks_per_node, -n $n_particles ($total_ranks MPI ranks)"
    for i in {1..1}; do
        srun -N 2 --ntasks-per-node=$tasks_per_node ./mpi -n $n_particles -s 1
    done
done
