
#include "common.h"
#include <mpi.h>
#include <cmath>
#include <algorithm>
#include <vector>

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

// 2D processor grid
int proc_grid_dims[2];          // Dimensions of the processor grid
int proc_grid_coords[2];        // Coordinates of this process in the grid
MPI_Comm cart_comm;             // Cartesian communicator

// Domain decomposition
double domain_x_min, domain_x_max;  // X boundaries for this process
double domain_y_min, domain_y_max;  // Y boundaries for this process
double domain_size_x, domain_size_y; // Dimensions of this process's domain

// Neighbor ranks in 2D grid (left, right, up, down, and diagonals)
int neighbors[8];

// Cell-based simulation data structure for faster force calculation
struct cell_t {
    std::vector<int> particles;  // Indices of particles in this cell
};

// Grid of cells
cell_t** grid = nullptr;
int grid_size = 0;
double cell_size = 0;

// Ghost particles from neighboring processes
std::vector<particle_t> ghost_particles;

// Local particles managed by this process
std::vector<particle_t> local_particles;

// Get cell coordinates for a particle
inline void get_cell_coords(double x, double y, int& cell_x, int& cell_y) {
    cell_x = std::min(static_cast<int>((x - domain_x_min) / cell_size), grid_size - 1);
    cell_y = std::min(static_cast<int>((y - domain_y_min) / cell_size), grid_size - 1);
    // Ensure cell indices are within bounds
    cell_x = std::max(0, std::min(cell_x, grid_size - 1));
    cell_y = std::max(0, std::min(cell_y, grid_size - 1));
}

// Add particle to the appropriate cell
void add_to_grid(int particle_idx, particle_t& p) {
    int cell_x, cell_y;
    get_cell_coords(p.x, p.y, cell_x, cell_y);
    grid[cell_x][cell_y].particles.push_back(particle_idx);
}

// Check if a particle is within this process's domain
bool is_in_domain(const particle_t& p) {
    return (p.x >= domain_x_min && p.x < domain_x_max &&
            p.y >= domain_y_min && p.y < domain_y_max);
}

// Check if a particle is within the ghost region (cutoff distance from domain boundary)
bool is_in_ghost_region(const particle_t& p) {
    // Calculate distance to each domain boundary
    double dist_to_x_min = fabs(p.x - domain_x_min);
    double dist_to_x_max = fabs(p.x - domain_x_max);
    double dist_to_y_min = fabs(p.y - domain_y_min);
    double dist_to_y_max = fabs(p.y - domain_y_max);
    
    // If particle is outside domain but within cutoff distance of a boundary
    return (!is_in_domain(p) && 
            ((p.x < domain_x_min && dist_to_x_min < cutoff) ||
             (p.x >= domain_x_max && dist_to_x_max < cutoff) ||
             (p.y < domain_y_min && dist_to_y_min < cutoff) ||
             (p.y >= domain_y_max && dist_to_y_max < cutoff)));
}

// Build the grid data structure
void build_grid() {
    // Clear the old grid
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            grid[i][j].particles.clear();
        }
    }
    
    // Add local particles to the grid
    for (size_t i = 0; i < local_particles.size(); i++) {
        add_to_grid(i, local_particles[i]);
    }
    
    // Add ghost particles to the grid
    for (size_t i = 0; i < ghost_particles.size(); i++) {
        // Ghost particles are indexed after local particles
        add_to_grid(local_particles.size() + i, ghost_particles[i]);
    }
}

// Setup 2D decomposition
void setup_2d_decomposition(int num_procs) {
    // Find a good 2D decomposition (try to make it as square as possible)
    proc_grid_dims[0] = proc_grid_dims[1] = 0;
    
    // Find factors of num_procs to determine grid dimensions
    int max_dim = sqrt(num_procs) + 1;
    for (int i = 1; i <= max_dim; i++) {
        if (num_procs % i == 0) {
            proc_grid_dims[0] = i;
            proc_grid_dims[1] = num_procs / i;
        }
    }
    
    // Create a Cartesian communicator
    int periods[2] = {0, 0}; // Non-periodic boundaries
    MPI_Cart_create(MPI_COMM_WORLD, 2, proc_grid_dims, periods, 1, &cart_comm);
    
    // Get coordinates of this process in the grid
    int my_rank;
    MPI_Comm_rank(cart_comm, &my_rank);
    MPI_Cart_coords(cart_comm, my_rank, 2, proc_grid_coords);
    
    // Find neighbor processes (including diagonals)
    // Order: left, right, up, down, up-left, up-right, down-left, down-right
    
    // Default to MPI_PROC_NULL for missing neighbors (boundary processes)
    for (int i = 0; i < 8; i++) {
        neighbors[i] = MPI_PROC_NULL;
    }
    
    // Calculate valid neighbors safely
    int coords[2];
    
    // Left neighbor
    if (proc_grid_coords[0] > 0) {
        coords[0] = proc_grid_coords[0] - 1;
        coords[1] = proc_grid_coords[1];
        MPI_Cart_rank(cart_comm, coords, &neighbors[0]);
    }
    
    // Right neighbor
    if (proc_grid_coords[0] < proc_grid_dims[0] - 1) {
        coords[0] = proc_grid_coords[0] + 1;
        coords[1] = proc_grid_coords[1];
        MPI_Cart_rank(cart_comm, coords, &neighbors[1]);
    }
    
    // Up neighbor
    if (proc_grid_coords[1] > 0) {
        coords[0] = proc_grid_coords[0];
        coords[1] = proc_grid_coords[1] - 1;
        MPI_Cart_rank(cart_comm, coords, &neighbors[2]);
    }
    
    // Down neighbor
    if (proc_grid_coords[1] < proc_grid_dims[1] - 1) {
        coords[0] = proc_grid_coords[0];
        coords[1] = proc_grid_coords[1] + 1;
        MPI_Cart_rank(cart_comm, coords, &neighbors[3]);
    }
    
    // Up-left neighbor
    if (proc_grid_coords[0] > 0 && proc_grid_coords[1] > 0) {
        coords[0] = proc_grid_coords[0] - 1;
        coords[1] = proc_grid_coords[1] - 1;
        MPI_Cart_rank(cart_comm, coords, &neighbors[4]);
    }
    
    // Up-right neighbor
    if (proc_grid_coords[0] < proc_grid_dims[0] - 1 && proc_grid_coords[1] > 0) {
        coords[0] = proc_grid_coords[0] + 1;
        coords[1] = proc_grid_coords[1] - 1;
        MPI_Cart_rank(cart_comm, coords, &neighbors[5]);
    }
    
    // Down-left neighbor
    if (proc_grid_coords[0] > 0 && proc_grid_coords[1] < proc_grid_dims[1] - 1) {
        coords[0] = proc_grid_coords[0] - 1;
        coords[1] = proc_grid_coords[1] + 1;
        MPI_Cart_rank(cart_comm, coords, &neighbors[6]);
    }
    
    // Down-right neighbor
    if (proc_grid_coords[0] < proc_grid_dims[0] - 1 && proc_grid_coords[1] < proc_grid_dims[1] - 1) {
        coords[0] = proc_grid_coords[0] + 1;
        coords[1] = proc_grid_coords[1] + 1;
        MPI_Cart_rank(cart_comm, coords, &neighbors[7]);
    }
}

// Forward declarations
void exchange_ghost_particles();

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Setup 2D process grid
    setup_2d_decomposition(num_procs);
    
    // Calculate domain boundaries for this process
    domain_size_x = size / proc_grid_dims[0];
    domain_size_y = size / proc_grid_dims[1];
    
    domain_x_min = proc_grid_coords[0] * domain_size_x;
    domain_x_max = domain_x_min + domain_size_x;
    domain_y_min = proc_grid_coords[1] * domain_size_y;
    domain_y_max = domain_y_min + domain_size_y;
    
    // Initialize the grid for spatial partitioning
    // We want cell_size to be slightly larger than cutoff
    cell_size = cutoff * 1.1;
    grid_size = std::max(
        static_cast<int>(domain_size_x / cell_size) + 1,
        static_cast<int>(domain_size_y / cell_size) + 1
    );
    
    // Allocate the grid
    grid = new cell_t*[grid_size];
    for (int i = 0; i < grid_size; i++) {
        grid[i] = new cell_t[grid_size];
    }
    
    // Initialize local particles
    local_particles.clear();
    for (int i = 0; i < num_parts; i++) {
        if (is_in_domain(parts[i])) {
            local_particles.push_back(parts[i]);
        }
    }
    
    // Initialize ghost particles (empty at start)
    ghost_particles.clear();
    
    // Exchange initial ghost particles
    exchange_ghost_particles();
    
    // Build initial grid
    build_grid();
}

// Exchange ghost particles with neighboring processes
void exchange_ghost_particles() {
    // Clear the current ghost particles
    ghost_particles.clear();
    
    // For each neighbor, identify particles to send (within cutoff of boundary)
    std::vector<std::vector<particle_t>> send_particles(8);
    
    // Collect particles near each boundary to send to neighbors
    for (const auto& p : local_particles) {
        // Check left boundary
        if (p.x - domain_x_min < cutoff && neighbors[0] != MPI_PROC_NULL) {
            send_particles[0].push_back(p);
        }
        // Check right boundary
        if (domain_x_max - p.x < cutoff && neighbors[1] != MPI_PROC_NULL) {
            send_particles[1].push_back(p);
        }
        // Check top boundary
        if (p.y - domain_y_min < cutoff && neighbors[2] != MPI_PROC_NULL) {
            send_particles[2].push_back(p);
        }
        // Check bottom boundary
        if (domain_y_max - p.y < cutoff && neighbors[3] != MPI_PROC_NULL) {
            send_particles[3].push_back(p);
        }
        
        // Check corners (only if particle is close to both corresponding boundaries)
        // Top-left
        if (p.x - domain_x_min < cutoff && p.y - domain_y_min < cutoff && neighbors[4] != MPI_PROC_NULL) {
            send_particles[4].push_back(p);
        }
        // Top-right
        if (domain_x_max - p.x < cutoff && p.y - domain_y_min < cutoff && neighbors[5] != MPI_PROC_NULL) {
            send_particles[5].push_back(p);
        }
        // Bottom-left
        if (p.x - domain_x_min < cutoff && domain_y_max - p.y < cutoff && neighbors[6] != MPI_PROC_NULL) {
            send_particles[6].push_back(p);
        }
        // Bottom-right
        if (domain_x_max - p.x < cutoff && domain_y_max - p.y < cutoff && neighbors[7] != MPI_PROC_NULL) {
            send_particles[7].push_back(p);
        }
    }
    
    // Prepare to send/receive particles
    MPI_Request requests[16];  // 8 sends + 8 receives
    int req_count = 0;
    
    // For each direction, send and receive in non-blocking mode
    for (int i = 0; i < 8; i++) {
        int neighbor_rank = neighbors[i];
        if (neighbor_rank != MPI_PROC_NULL) {
            // Send particles to neighbor
            if (!send_particles[i].empty()) {
                MPI_Isend(send_particles[i].data(), send_particles[i].size() * sizeof(particle_t),
                         MPI_BYTE, neighbor_rank, 0, cart_comm, &requests[req_count++]);
            } else {
                // Send empty message to avoid deadlock
                MPI_Isend(NULL, 0, MPI_BYTE, neighbor_rank, 0, cart_comm, &requests[req_count++]);
            }
            
            // Probe for message size
            MPI_Status status;
            MPI_Probe(neighbor_rank, 0, cart_comm, &status);
            
            int msg_size;
            MPI_Get_count(&status, MPI_BYTE, &msg_size);
            int num_particles = msg_size / sizeof(particle_t);
            
            if (num_particles > 0) {
                // Allocate buffer for receiving
                std::vector<particle_t> recv_buffer(num_particles);
                
                // Receive particles from neighbor
                MPI_Irecv(recv_buffer.data(), msg_size, MPI_BYTE,
                         neighbor_rank, 0, cart_comm, &requests[req_count++]);
                
                // Append to ghost particles once received
                ghost_particles.insert(ghost_particles.end(), recv_buffer.begin(), recv_buffer.end());
            } else {
                // Receive empty message
                MPI_Irecv(NULL, 0, MPI_BYTE, neighbor_rank, 0, cart_comm, &requests[req_count++]);
            }
        }
    }
    
    // Wait for all sends/receives to complete
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
}

// Exchange particles that have moved to other domains
void exchange_moved_particles() {
    // Identify particles that have moved outside our domain
    std::vector<particle_t> outgoing_particles[8];
    std::vector<int> indices_to_remove;
    
    for (size_t i = 0; i < local_particles.size(); i++) {
        particle_t& p = local_particles[i];
        
        // Determine which neighbor domain this particle belongs to now
        if (p.x < domain_x_min) {
            // Left side
            if (p.y < domain_y_min) {
                // Top-left corner
                if (neighbors[4] != MPI_PROC_NULL) {
                    outgoing_particles[4].push_back(p);
                    indices_to_remove.push_back(i);
                }
            } else if (p.y >= domain_y_max) {
                // Bottom-left corner
                if (neighbors[6] != MPI_PROC_NULL) {
                    outgoing_particles[6].push_back(p);
                    indices_to_remove.push_back(i);
                }
            } else {
                // Left edge
                if (neighbors[0] != MPI_PROC_NULL) {
                    outgoing_particles[0].push_back(p);
                    indices_to_remove.push_back(i);
                }
            }
        } else if (p.x >= domain_x_max) {
            // Right side
            if (p.y < domain_y_min) {
                // Top-right corner
                if (neighbors[5] != MPI_PROC_NULL) {
                    outgoing_particles[5].push_back(p);
                    indices_to_remove.push_back(i);
                }
            } else if (p.y >= domain_y_max) {
                // Bottom-right corner
                if (neighbors[7] != MPI_PROC_NULL) {
                    outgoing_particles[7].push_back(p);
                    indices_to_remove.push_back(i);
                }
            } else {
                // Right edge
                if (neighbors[1] != MPI_PROC_NULL) {
                    outgoing_particles[1].push_back(p);
                    indices_to_remove.push_back(i);
                }
            }
        } else if (p.y < domain_y_min) {
            // Top edge (not corners)
            if (neighbors[2] != MPI_PROC_NULL) {
                outgoing_particles[2].push_back(p);
                indices_to_remove.push_back(i);
            }
        } else if (p.y >= domain_y_max) {
            // Bottom edge (not corners)
            if (neighbors[3] != MPI_PROC_NULL) {
                outgoing_particles[3].push_back(p);
                indices_to_remove.push_back(i);
            }
        }
    }
    
    // Prepare to send/receive particles
    MPI_Request requests[16];  // 8 sends + 8 receives
    std::vector<std::vector<particle_t>> incoming_particles(8);
    int req_count = 0;
    
    // For each direction, send and receive in non-blocking mode
    for (int i = 0; i < 8; i++) {
        int neighbor_rank = neighbors[i];
        if (neighbor_rank != MPI_PROC_NULL) {
            // Send particles to neighbor
            if (!outgoing_particles[i].empty()) {
                MPI_Isend(outgoing_particles[i].data(), outgoing_particles[i].size() * sizeof(particle_t),
                         MPI_BYTE, neighbor_rank, 1, cart_comm, &requests[req_count++]);
            } else {
                // Send empty message to avoid deadlock
                MPI_Isend(NULL, 0, MPI_BYTE, neighbor_rank, 1, cart_comm, &requests[req_count++]);
            }
            
            // Probe for message size
            MPI_Status status;
            MPI_Probe(neighbor_rank, 1, cart_comm, &status);
            
            int msg_size;
            MPI_Get_count(&status, MPI_BYTE, &msg_size);
            int num_particles = msg_size / sizeof(particle_t);
            
            if (num_particles > 0) {
                // Allocate buffer for receiving
                incoming_particles[i].resize(num_particles);
                
                // Receive particles from neighbor
                MPI_Irecv(incoming_particles[i].data(), msg_size, MPI_BYTE,
                         neighbor_rank, 1, cart_comm, &requests[req_count++]);
            } else {
                // Receive empty message
                MPI_Irecv(NULL, 0, MPI_BYTE, neighbor_rank, 1, cart_comm, &requests[req_count++]);
            }
        }
    }
    
    // Wait for all sends/receives to complete
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    
    // Remove particles that moved to other domains
    std::sort(indices_to_remove.begin(), indices_to_remove.end(), std::greater<int>());
    for (int idx : indices_to_remove) {
        local_particles.erase(local_particles.begin() + idx);
    }
    
    // Add incoming particles to local domain
    for (int i = 0; i < 8; i++) {
        for (const auto& p : incoming_particles[i]) {
            if (is_in_domain(p)) {
                local_particles.push_back(p);
            }
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Reset acceleration for all local particles
    for (auto& p : local_particles) {
        p.ax = 0;
        p.ay = 0;
    }
    
    // Compute forces between particles using the grid
    for (size_t i = 0; i < local_particles.size(); i++) {
        particle_t& local_particle = local_particles[i];
        int p_cell_x, p_cell_y;
        get_cell_coords(local_particle.x, local_particle.y, p_cell_x, p_cell_y);
        
        // Check the particle's cell and all neighboring cells
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int neighbor_x = p_cell_x + dx;
                int neighbor_y = p_cell_y + dy;
                
                // Skip cells outside the grid
                if (neighbor_x < 0 || neighbor_x >= grid_size || 
                    neighbor_y < 0 || neighbor_y >= grid_size)
                    continue;
                
                // Check all particles in the neighboring cell
                for (int idx : grid[neighbor_x][neighbor_y].particles) {
                    // If index points to a local particle
                    if (idx < (int)local_particles.size()) {
                        // Skip self-interaction
                        if (i != (size_t)idx) {
                            apply_force(local_particle, local_particles[idx]);
                        }
                    }
                    // If index points to a ghost particle
                    else {
                        int ghost_idx = idx - local_particles.size();
                        apply_force(local_particle, ghost_particles[ghost_idx]);
                    }
                }
            }
        }
    }
    
    // Move all local particles
    for (auto& p : local_particles) {
        move(p, size);
    }
    
    // Exchange particles that moved to other domains
    exchange_moved_particles();
    
    // Update ghost particles
    exchange_ghost_particles();
    
    // Rebuild the grid with updated particle positions
    build_grid();
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Count total particles across all processes
    int local_count = local_particles.size();
    int* counts = new int[num_procs];
    int* displs = new int[num_procs];
    
    // Gather counts of particles on each process
    MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Only rank 0 computes displacements and gathers particles
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < num_procs; i++) {
            displs[i] = displs[i-1] + counts[i-1];
        }
    }
    
    // Gather all particles to process 0
    MPI_Gatherv(local_particles.data(), local_count, PARTICLE,
               parts, counts, displs, PARTICLE, 0, MPI_COMM_WORLD);
    
    // Sort the particles by ID on the master process for saving
    if (rank == 0) {
        std::sort(parts, parts + num_parts, 
            [](const particle_t& a, const particle_t& b) { return a.id < b.id; });
    }
    
    delete[] counts;
    delete[] displs;
}

// Clean up resources
void finalize_simulation() {
    // Free the grid
    if (grid != nullptr) {
        for (int i = 0; i < grid_size; i++) {
            delete[] grid[i];
        }
        delete[] grid;
        grid = nullptr;
    }
    
    // Free MPI communicator
    MPI_Comm_free(&cart_comm);
}
