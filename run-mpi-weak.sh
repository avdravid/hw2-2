#!/bin/bash
make

# Define ratios for particles per rank (work per rank)
ratios=(1000 2000 5000 10000)

# Number of tasks per node for scaling (same as before)
tasks_per_node_values=(1 2 4 8 16 32 64)

# Loop through each ratio
for particles_per_rank in "${ratios[@]}"; do
    echo "=== Running weak scaling tests for particles per rank = $particles_per_rank ==="

    # Special case: 1 rank (1 node, 1 task per node)
    total_ranks=1
    n_particles=$((total_ranks * particles_per_rank))
    echo "Running weak scaling with -N 1 --ntasks-per-node=1, -n $n_particles (1 MPI rank)"
    for i in {1..1}; do
        srun -N 1 --ntasks-per-node=1 ./mpi -n $n_particles -s 1
    done

    # Remaining cases: 2 nodes with increasing ranks (2, 4, 8, 16, 32, 64, 128 ranks)
    for tasks_per_node in "${tasks_per_node_values[@]}"; do
        total_ranks=$((2 * tasks_per_node))
        n_particles=$((total_ranks * particles_per_rank))  # Total particles increases with rank count

        echo "Running weak scaling with -N 2 --ntasks-per-node=$tasks_per_node, -n $n_particles ($total_ranks MPI ranks)"
        for i in {1..1}; do
            srun -N 2 --ntasks-per-node=$tasks_per_node ./mpi -n $n_particles -s 1
        done
    done

    echo "=== Finished weak scaling for particles per rank = $particles_per_rank ==="
    echo
done

