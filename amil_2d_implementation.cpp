#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>

// Define structure for 2D domain information
typedef struct {
    double x_min;  // Lower bound of this domain's x-range
    double x_max;  // Upper bound of this domain's x-range
    double y_min;  // Lower bound of this domain's y-range
    double y_max;  // Upper bound of this domain's y-range
    double cutoff_boundary; // Extended boundary for force calculation
} domain_t;

// Grid dimensions for 2D decomposition
int grid_size_x;
int grid_size_y;

// Apply the force from neighbor to particle
void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    
    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;
    
    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);
    
    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    
    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

// Get the grid coordinates (i,j) from rank
void rank_to_grid(int rank, int& grid_i, int& grid_j) {
    grid_i = rank / grid_size_y;
    grid_j = rank % grid_size_y;
}

// Get rank from grid coordinates
int grid_to_rank(int grid_i, int grid_j) {
    return grid_i * grid_size_y + grid_j;
}

// Initialize simulation with 2D domain decomposition
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Calculate grid dimensions for a roughly square decomposition
    grid_size_x = static_cast<int>(sqrt(num_procs));
    while (num_procs % grid_size_x != 0) {
        grid_size_x--;  // Find largest divisor <= sqrt
    }
    grid_size_y = num_procs / grid_size_x;
    
    // Broadcast grid dimensions to ensure all processors have the same values
    int grid_dims[2] = {grid_size_x, grid_size_y};
    MPI_Bcast(grid_dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
    grid_size_x = grid_dims[0];
    grid_size_y = grid_dims[1];
}

// Get the domain assigned to a specific rank
domain_t get_domain(double size, int rank, int num_procs) {
    domain_t domain;
    
    // Convert rank to grid coordinates
    int grid_i, grid_j;
    rank_to_grid(rank, grid_i, grid_j);
    
    // Calculate domain boundaries
    double cell_width = size / grid_size_x;
    double cell_height = size / grid_size_y;
    
    domain.x_min = grid_i * cell_width;
    domain.x_max = (grid_i + 1) * cell_width;
    domain.y_min = grid_j * cell_height;
    domain.y_max = (grid_j + 1) * cell_height;
    domain.cutoff_boundary = cutoff;
    
    return domain;
}

// Check if a particle belongs to this processor's domain
bool is_in_domain(particle_t& p, domain_t& domain) {
    return (p.x >= domain.x_min && p.x < domain.x_max &&
            p.y >= domain.y_min && p.y < domain.y_max);
}

// Check if a particle is within the extended boundary (for ghost particles)
bool is_in_extended_domain(particle_t& p, domain_t& domain, double cutoff_boundary) {
    return (p.x >= domain.x_min - cutoff_boundary && p.x < domain.x_max + cutoff_boundary &&
            p.y >= domain.y_min - cutoff_boundary && p.y < domain.y_max + cutoff_boundary);
}

// Simulate one time step with 2D domain decomposition
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Get this processor's domain
    domain_t my_domain = get_domain(size, rank, num_procs);
    
    // Vectors to hold local particles and ghost particles
    std::vector<particle_t> my_particles;      // Particles that belong to this domain
    std::vector<particle_t> ghost_particles;   // Particles from neighbor domains needed for force calc
    
    // First, determine which particles belong to this processor and which are ghosts
    for (int i = 0; i < num_parts; i++) {
        if (is_in_domain(parts[i], my_domain)) {
            // This particle belongs to my domain
            my_particles.push_back(parts[i]);
        } 
        else if (is_in_extended_domain(parts[i], my_domain, my_domain.cutoff_boundary)) {
            // This is a ghost particle (needed for force calculation but owned by another processor)
            ghost_particles.push_back(parts[i]);
        }
    }
    
    // Clear accelerations for my particles
    for (auto& p : my_particles) {
        p.ax = p.ay = 0;
    }
    
    // Calculate forces between my particles
    for (int i = 0; i < my_particles.size(); i++) {
        for (int j = i + 1; j < my_particles.size(); j++) {
            apply_force(my_particles[i], my_particles[j]);
            apply_force(my_particles[j], my_particles[i]); // Newton's 3rd law
        }
    }
    
    // Calculate forces between my particles and ghost particles
    for (auto& my_p : my_particles) {
        for (auto& ghost_p : ghost_particles) {
            apply_force(my_p, ghost_p);
            // We don't apply force to ghost particles as they're handled by their owner processor
        }
    }
    
    // Move my particles
    for (auto& p : my_particles) {
        move(p, size);
    }
    
    // Determine which neighboring domain each particle belongs to
    std::vector<std::vector<particle_t>> outgoing_particles(num_procs);
    std::vector<particle_t> remaining_particles;
    
    // Convert rank to grid coordinates
    int my_i, my_j;
    rank_to_grid(rank, my_i, my_j);
    
    for (auto& p : my_particles) {
        if (is_in_domain(p, my_domain)) {
            remaining_particles.push_back(p);
        } else {
            // Figure out which domain this particle belongs to now
            int target_i = my_i;
            int target_j = my_j;
            
            // Determine direction based on particle position
            if (p.x < my_domain.x_min) target_i = (my_i - 1 + grid_size_x) % grid_size_x;
            else if (p.x >= my_domain.x_max) target_i = (my_i + 1) % grid_size_x;
            
            if (p.y < my_domain.y_min) target_j = (my_j - 1 + grid_size_y) % grid_size_y;
            else if (p.y >= my_domain.y_max) target_j = (my_j + 1) % grid_size_y;
            
            int target_rank = grid_to_rank(target_i, target_j);
            outgoing_particles[target_rank].push_back(p);
        }
    }
    
    // Exchange counts with all processes
    std::vector<int> send_counts(num_procs, 0);
    for (int i = 0; i < num_procs; i++) {
        send_counts[i] = outgoing_particles[i].size();
    }
    
    std::vector<int> recv_counts(num_procs, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Calculate displacements
    std::vector<int> send_displs(num_procs, 0);
    std::vector<int> recv_displs(num_procs, 0);
    
    for (int i = 1; i < num_procs; i++) {
        send_displs[i] = send_displs[i-1] + send_counts[i-1];
        recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
    }
    
    // Flatten outgoing particles into a single buffer
    int total_send = send_displs[num_procs-1] + send_counts[num_procs-1];
    int total_recv = recv_displs[num_procs-1] + recv_counts[num_procs-1];
    
    std::vector<particle_t> send_buffer(total_send);
    std::vector<particle_t> recv_buffer(total_recv);
    
    for (int i = 0; i < num_procs; i++) {
        for (int j = 0; j < send_counts[i]; j++) {
            send_buffer[send_displs[i] + j] = outgoing_particles[i][j];
        }
    }
    
    // Exchange particles
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), PARTICLE,
                 recv_buffer.data(), recv_counts.data(), recv_displs.data(), PARTICLE,
                 MPI_COMM_WORLD);
    
    // Merge received particles with remaining ones
    my_particles = remaining_particles;
    my_particles.insert(my_particles.end(), recv_buffer.begin(), recv_buffer.end());
    
    // Each processor needs to tell all others where its particles are for global indexing
    std::vector<int> particle_counts(num_procs, 0);
    particle_counts[rank] = my_particles.size();
    
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                 particle_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    
    // Calculate displacements for gathering
    std::vector<int> gather_displs(num_procs, 0);
    for (int i = 1; i < num_procs; i++) {
        gather_displs[i] = gather_displs[i-1] + particle_counts[i-1];
    }
    
    // Gather all particles to all processors
    MPI_Allgatherv(my_particles.data(), my_particles.size(), PARTICLE,
                  parts, particle_counts.data(), gather_displs.data(), PARTICLE,
                  MPI_COMM_WORLD);
    
    // No sorting here - only sort when saving output
}

// Gather particles to rank 0 for saving
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Particles are already gathered in simulate_one_step
    // Just ensure they're sorted consistently by ID
    std::sort(parts, parts + num_parts, 
             [](const particle_t& a, const particle_t& b) { return a.id < b.id; });
}
