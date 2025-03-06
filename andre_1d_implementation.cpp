#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <unordered_map>

// Define structure for 1D domain information
typedef struct {
    double x_min;  // Lower bound of this domain's x-range
    double x_max;  // Upper bound of this domain's x-range
    double cutoff_boundary; // Extended boundary for force calculation
} domain_t;

// Apply the force from neighbor to particle
// Inline for better performance
inline void apply_force(particle_t& particle, particle_t& neighbor) {
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
// Inline for better performance
inline void move(particle_t& p, double size) {
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

// Get the domain assigned to a specific rank
domain_t get_domain(double size, int rank, int num_procs) {
    domain_t domain;

    // Calculate 1D domain boundaries
    double cell_width = size / num_procs;

    domain.x_min = rank * cell_width;
    domain.x_max = (rank + 1) * cell_width;
    domain.cutoff_boundary = cutoff;

    return domain;
}

// Check if a particle belongs to this processor's domain
inline bool is_in_domain(const particle_t& p, const domain_t& domain) {
    return (p.x >= domain.x_min && p.x < domain.x_max);
}

// Check if a particle is within the extended boundary (for ghost particles)
inline bool is_in_extended_domain(const particle_t& p, const domain_t& domain, double cutoff_boundary) {
    return (p.x >= domain.x_min - cutoff_boundary && p.x < domain.x_max + cutoff_boundary);
}

// Initialize simulation with 1D domain decomposition
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Nothing special needed for 1D decomposition initialization
}

// Simulate one time step with 1D domain decomposition
void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Get this processor's domain
    domain_t my_domain = get_domain(size, rank, num_procs);
    
    // Reserve space for local and ghost particles to avoid reallocation
    std::vector<particle_t> my_particles;
    std::vector<particle_t> ghost_particles;
    
    // Pre-allocate with estimated sizes to avoid reallocation
    my_particles.reserve(num_parts / num_procs + 50);  // Add buffer for imbalance
    ghost_particles.reserve(num_parts / num_procs / 2 + 50);  // Estimate ghost particles
    
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
    #pragma omp parallel for
    for (size_t i = 0; i < my_particles.size(); i++) {
        my_particles[i].ax = 0;
        my_particles[i].ay = 0;
    }

    // Use spatial binning for force calculation to reduce complexity
    // Create bins for particles to enable O(n) force calculation instead of O(n²)
    const double bin_size = cutoff;
    const int bins_x = ceil(size / bin_size);
    const int bins_y = ceil(size / bin_size);
    
    // Hash function for bin coordinates
    auto bin_hash = [bins_y](int bin_x, int bin_y) {
        return bin_x * bins_y + bin_y;
    };
    
    // Map bins to particles
    std::unordered_map<int, std::vector<int>> binned_particles;
    std::unordered_map<int, std::vector<int>> binned_ghosts;
    
    // Bin local particles
    for (size_t i = 0; i < my_particles.size(); i++) {
        int bin_x = my_particles[i].x / bin_size;
        int bin_y = my_particles[i].y / bin_size;
        binned_particles[bin_hash(bin_x, bin_y)].push_back(i);
    }
    
    // Bin ghost particles
    for (size_t i = 0; i < ghost_particles.size(); i++) {
        int bin_x = ghost_particles[i].x / bin_size;
        int bin_y = ghost_particles[i].y / bin_size;
        binned_ghosts[bin_hash(bin_x, bin_y)].push_back(i);
    }
    
    // Compute forces using spatial binning
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < my_particles.size(); i++) {
            particle_t& p = my_particles[i];
            int bin_x = p.x / bin_size;
            int bin_y = p.y / bin_size;
            
            // Check particle's bin and neighboring bins
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    int nb_x = bin_x + dx;
                    int nb_y = bin_y + dy;
                    
                    // Skip invalid bins
                    if (nb_x < 0 || nb_x >= bins_x || nb_y < 0 || nb_y >= bins_y)
                        continue;
                    
                    int bin_idx = bin_hash(nb_x, nb_y);
                    
                    // Apply forces from local particles in this bin
                    auto it_local = binned_particles.find(bin_idx);
                    if (it_local != binned_particles.end()) {
                        for (int j : it_local->second) {
                            if (i != j) {  // Skip self-interaction
                                apply_force(p, my_particles[j]);
                            }
                        }
                    }
                    
                    // Apply forces from ghost particles in this bin
                    auto it_ghost = binned_ghosts.find(bin_idx);
                    if (it_ghost != binned_ghosts.end()) {
                        for (int j : it_ghost->second) {
                            apply_force(p, ghost_particles[j]);
                        }
                    }
                }
            }
        }
    }
    
    // Move my particles
    #pragma omp parallel for
    for (size_t i = 0; i < my_particles.size(); i++) {
        move(my_particles[i], size);
    }

    // Determine particles to send to neighbors
    std::vector<std::vector<particle_t>> outgoing_particles(num_procs);
    std::vector<particle_t> remaining_particles;
    
    // Pre-allocate to avoid reallocation
    remaining_particles.reserve(my_particles.size());
    for (int i = 0; i < num_procs; i++) {
        outgoing_particles[i].reserve(50);  // Small buffer for particles moving between domains
    }
    
    for (auto& p : my_particles) {
        if (is_in_domain(p, my_domain)) {
            remaining_particles.push_back(p);
        } else {
            // Find new domain for this particle
            int target_rank = std::min(std::max(static_cast<int>(p.x * num_procs / size), 0), num_procs - 1);
            outgoing_particles[target_rank].push_back(p);
        }
    }

    // Exchange counts with all processes
    std::vector<int> send_counts(num_procs);
    for (int i = 0; i < num_procs; i++) {
        send_counts[i] = outgoing_particles[i].size();
    }

    std::vector<int> recv_counts(num_procs);
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
        // Copy in parallel for large datasets
        #pragma omp parallel for if(send_counts[i] > 1000)
        for (int j = 0; j < send_counts[i]; j++) {
            send_buffer[send_displs[i] + j] = outgoing_particles[i][j];
        }
    }

    // Exchange particles
    MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(), PARTICLE,
                  recv_buffer.data(), recv_counts.data(), recv_displs.data(), PARTICLE,
                  MPI_COMM_WORLD);

    // Use non-blocking communication for ghost particles to overlap computation and communication
    // Prepare buffers for left and right neighbors' ghost particles
    std::vector<particle_t> left_send_buffer;
    std::vector<particle_t> right_send_buffer;
    
    // Only send ghost particles to immediate neighbors in 1D decomposition
    int left_neighbor = (rank > 0) ? rank - 1 : num_procs - 1;
    int right_neighbor = (rank < num_procs - 1) ? rank + 1 : 0;
    
    // Collect ghost particles for left neighbor
    for (auto& p : remaining_particles) {
        if (p.x < my_domain.x_min + cutoff && p.x >= my_domain.x_min) {
            left_send_buffer.push_back(p);
        }
    }
    
    // Collect ghost particles for right neighbor
    for (auto& p : remaining_particles) {
        if (p.x >= my_domain.x_max - cutoff && p.x < my_domain.x_max) {
            right_send_buffer.push_back(p);
        }
    }
    
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
}

// Gather particles to rank 0 for saving
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Sort by ID for consistent output
    std::sort(parts, parts + num_parts,
             [](const particle_t& a, const particle_t& b) { return a.id < b.id; });
}
