#!/bin/bash

# Compile the code
make

# Set node and task configuration
NODES=1
TASKS_PER_NODE=1
TOTAL_TASKS=$((NODES * TASKS_PER_NODE))

# Define particle counts to iterate over
particles=(1000 10000 100000 1000000 6000000)

# Run srun for each particle count, repeating 3 times
for n in "${particles[@]}"; do
    for i in {1..1}; do
        echo "Running with -N $NODES, --ntasks-per-node=$TASKS_PER_NODE, -n $n (Iteration $i)"
        srun -N $NODES --ntasks-per-node=$TASKS_PER_NODE ./mpi -n "$n" -s 1
    done
done

