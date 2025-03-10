
#include "common.h"       
#include <mpi.h>          
#include <cmath>         
#include <algorithm>       
#include <vector>          

//Function to calculate and apply the repulsive force between two particles
void apply_force(particle_t& particle, particle_t& neighbor) {

    //this will calculate the distance between the particle and its neighbor in x and y directions
    double distance_x = neighbor.x - particle.x;
    double distance_y = neighbor.y - particle.y;

    //sqr distance
    double squared_distance = distance_x * distance_x + distance_y * distance_y;

    //cutoff distance squared -> no force is applied
    if (squared_distance > cutoff * cutoff) return;
    
    squared_distance = fmax(squared_distance, min_r * min_r);
    double distance = sqrt(squared_distance);
    double coefficient = (1 - cutoff / distance) / squared_distance / mass;

    //Add the force to the particle's acceleration in x and y directions
    particle.ax += coefficient * distance_x;
    particle.ay += coefficient * distance_y;
}

//method that updates a particle's position and velocity based on acceleration
void move(particle_t& particle, double size) {

    // velocity using current acceleration and time step 
    particle.vx += particle.ax * dt;
    particle.vy += particle.ay * dt;

    //new pos using new velocity and time step
    particle.x += particle.vx * dt;
    particle.y += particle.vy * dt;

    // keep particle within "walls"
    while (particle.x < 0 || particle.x > size) {
  
        particle.x = particle.x < 0 ? -particle.x : 2 * size - particle.x;
        particle.vx = -particle.vx; 
    }
 // Same thing but for y direction
    while (particle.y < 0 || particle.y > size) {
    
        particle.y = particle.y < 0 ? -particle.y : 2 * size - particle.y;
        particle.vy = -particle.vy; 
    }
}

// Vars for2D grid
int proc_grid_dims[2];     //dims of processor grid (row, cols)
int proc_grid_coords[2];   // Coords

MPI_Comm cart_comm;        //MPI for the 2D grid

//vars for the domain the processor handles
double domain_x_min, domain_x_max; 
double domain_y_min, domain_y_max; 
double domain_size_x, domain_size_y; 

//data struct to store ranks of neighboring processors (all 8 directions)
int neighbors[8];

//Struct to hold a particle and its cell index
struct particle_with_cell {
    particle_t p;          
    int cell_idx;          
};

//cell as flattened grid
struct flat_cell_t {
    int start;            
    int count;            
};

std::vector<flat_cell_t> flat_grid;      
std::vector<int> particle_indices;       
std::vector<particle_with_cell> local_particles; 
std::vector<particle_t> ghost_particles;
int grid_size = 0;                       
double cell_size = 0;                   

//calcukate the 1D cell index for a particle based on its pos 
inline int get_cell_idx(double x, double y) {
    // cell coords within bounds

    int cell_x = std::min(static_cast<int>((x - domain_x_min) / cell_size), grid_size - 1);
    int cell_y = std::min(static_cast<int>((y - domain_y_min) / cell_size), grid_size - 1);
    cell_x = std::max(0, cell_x);
    cell_y = std::max(0, cell_y);

    //2d to 1d
    return cell_y * grid_size + cell_x;
}

//Bool check if particle is within processor domain
bool is_in_domain(const particle_t& particle) {

    return (particle.x >= domain_x_min && particle.x < domain_x_max &&
            particle.y >= domain_y_min && particle.y < domain_y_max);
}

//build the grid 
void build_grid() {
    //clear the old list of particle
    particle_indices.clear();

    //reserve space for all local and ghost particles to avoid resizing
    particle_indices.reserve(local_particles.size() + ghost_particles.size());
    
    //set count to 0 again
    for (auto& cell : flat_grid) {
        cell.count = 0;
    }
    
    //count how many particles are in each cell
    for (size_t i = 0; i < local_particles.size(); i++) {
        int new_idx = get_cell_idx(local_particles[i].p.x, local_particles[i].p.y);
        local_particles[i].cell_idx = new_idx;
        flat_grid[new_idx].count++;           
    }
    for (size_t i = 0; i < ghost_particles.size(); i++) {
        int idx = get_cell_idx(ghost_particles[i].x, ghost_particles[i].y);
        flat_grid[idx].count++;               
    }
    
    //sets pos for each cell in the particle_indices array
    int offset = 0;
    for (auto& cell : flat_grid) {
        cell.start = offset;
        offset += cell.count;               
    }
    
    // Now we fill the particle_indices array with particle indices
    std::vector<int> current_counts(flat_grid.size(), 0);
    for (size_t i = 0; i < local_particles.size(); i++) {
        int idx = local_particles[i].cell_idx;
        particle_indices[flat_grid[idx].start + current_counts[idx]] = i;
        current_counts[idx]++;                
    }
    for (size_t i = 0; i < ghost_particles.size(); i++) {
        int idx = get_cell_idx(ghost_particles[i].x, ghost_particles[i].y);
        particle_indices[flat_grid[idx].start + current_counts[idx]] = local_particles.size() + i;
        current_counts[idx]++;               
    }
}

//set up the 2D grid of processors
void setup_2d_decomposition(int num_procs) {

    int max_dim = sqrt(num_procs) + 1;
    for (int i = 1; i <= max_dim; i++) {
        if (num_procs % i == 0) {
            proc_grid_dims[0] = i;            //rows
            proc_grid_dims[1] = num_procs / i; //cols
        }
    }


    int periods[2] = {0, 0};  

    //create a 2D communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, proc_grid_dims, periods, 1, &cart_comm);
    int my_rank;
    MPI_Comm_rank(cart_comm, &my_rank);   // processor's rank
    MPI_Cart_coords(cart_comm, my_rank, 2, proc_grid_coords); //get coordinates
    
    //set all neighbor ranks to null (no neigh) initially
    for (int i = 0; i < 8; i++) neighbors[i] = MPI_PROC_NULL;
    int coords[2];
    //we then set the ranks of each neighbor
    if (proc_grid_coords[0] > 0) {
        coords[0] = proc_grid_coords[0] - 1; coords[1] = proc_grid_coords[1];
        MPI_Cart_rank(cart_comm, coords, &neighbors[0]);
    }

    if (proc_grid_coords[0] < proc_grid_dims[0] - 1) {
        coords[0] = proc_grid_coords[0] + 1; coords[1] = proc_grid_coords[1];
        MPI_Cart_rank(cart_comm, coords, &neighbors[1]);
    }
  
    if (proc_grid_coords[1] > 0) {
        coords[0] = proc_grid_coords[0]; coords[1] = proc_grid_coords[1] - 1;
        MPI_Cart_rank(cart_comm, coords, &neighbors[2]);
    }
  
    if (proc_grid_coords[1] < proc_grid_dims[1] - 1) {
        coords[0] = proc_grid_coords[0]; coords[1] = proc_grid_coords[1] + 1;
        MPI_Cart_rank(cart_comm, coords, &neighbors[3]);
    }
  
    if (proc_grid_coords[0] > 0 && proc_grid_coords[1] > 0) {
        coords[0] = proc_grid_coords[0] - 1; coords[1] = proc_grid_coords[1] - 1;
        MPI_Cart_rank(cart_comm, coords, &neighbors[4]);
    }

    if (proc_grid_coords[0] < proc_grid_dims[0] - 1 && proc_grid_coords[1] > 0) {
        coords[0] = proc_grid_coords[0] + 1; coords[1] = proc_grid_coords[1] - 1;
        MPI_Cart_rank(cart_comm, coords, &neighbors[5]);
    }

    if (proc_grid_coords[0] > 0 && proc_grid_coords[1] < proc_grid_dims[1] - 1) {
        coords[0] = proc_grid_coords[0] - 1; coords[1] = proc_grid_coords[1] + 1;
        MPI_Cart_rank(cart_comm, coords, &neighbors[6]);
    }
   
    if (proc_grid_coords[0] < proc_grid_dims[0] - 1 && proc_grid_coords[1] < proc_grid_dims[1] - 1) {
        coords[0] = proc_grid_coords[0] + 1; coords[1] = proc_grid_coords[1] + 1;
        MPI_Cart_rank(cart_comm, coords, &neighbors[7]);
    }
}

//share particles near domain boundaries with neighboring processors
void exchange_ghost_particles() {
    ghost_particles.clear(); //clear any old info
    //send and recieve the info
    std::vector<std::vector<particle_t>> send_particles(8, std::vector<particle_t>());
    std::vector<std::vector<particle_t>> recv_buffers(8, std::vector<particle_t>());


    //we preallocate the space to avoid resizing during the loop
    for (int i = 0; i < 8; i++) {
        send_particles[i].reserve(local_particles.size() / 10);
        recv_buffers[i].reserve(local_particles.size() / 10);
    }

    // Identify which particles we communicate to neighbors and whiuch ones
    for (const auto& pwc : local_particles) {


        const particle_t& p = pwc.p;

        if (p.x - domain_x_min < cutoff && neighbors[0] != MPI_PROC_NULL) send_particles[0].push_back(p);

        if (domain_x_max - p.x < cutoff && neighbors[1] != MPI_PROC_NULL) send_particles[1].push_back(p);
    
        if (p.y - domain_y_min < cutoff && neighbors[2] != MPI_PROC_NULL) send_particles[2].push_back(p);
        
        if (domain_y_max - p.y < cutoff && neighbors[3] != MPI_PROC_NULL) send_particles[3].push_back(p);
        
        if (p.x - domain_x_min < cutoff && p.y - domain_y_min < cutoff && neighbors[4] != MPI_PROC_NULL) send_particles[4].push_back(p);
        
        if (domain_x_max - p.x < cutoff && p.y - domain_y_min < cutoff && neighbors[5] != MPI_PROC_NULL) send_particles[5].push_back(p);
        
        if (p.x - domain_x_min < cutoff && domain_y_max - p.y < cutoff && neighbors[6] != MPI_PROC_NULL) send_particles[6].push_back(p);
        
        if (domain_x_max - p.x < cutoff && domain_y_max - p.y < cutoff && neighbors[7] != MPI_PROC_NULL) send_particles[7].push_back(p);
    }

    MPI_Request requests[16]; //MPI communication requests
    int req_count = 0;        //num of active requests
    int send_counts[8], recv_counts[8]; //num particles to both send and recieve
    
    //first, er exchange the number of particles to be sent/rec
    for (int i = 0; i < 8; i++) {
        send_counts[i] = send_particles[i].size();
        if (neighbors[i] != MPI_PROC_NULL) {
            MPI_Isend(&send_counts[i], 1, MPI_INT, neighbors[i], 0, cart_comm, &requests[req_count++]);
            MPI_Irecv(&recv_counts[i], 1, MPI_INT, neighbors[i], 0, cart_comm, &requests[req_count++]);
        } 
        
        else {
            recv_counts[i] = 0; // No neighbor, no particles
        }
    }

    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE); //we wait for counts to be exchanged to avoid any deadlocks or race conditions
    req_count = 0;

    //after synch we exchange particles
    for (int i = 0; i < 8; i++) {
        if (neighbors[i] != MPI_PROC_NULL) {
            if (send_counts[i] > 0) {
                MPI_Isend(send_particles[i].data(), send_counts[i], PARTICLE, neighbors[i], 1, cart_comm, &requests[req_count++]);
            }
            if (recv_counts[i] > 0) {
                recv_buffers[i].resize(recv_counts[i]);
                MPI_Irecv(recv_buffers[i].data(), recv_counts[i], PARTICLE, neighbors[i], 1, cart_comm, &requests[req_count++]);
            }
        }
    }
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE); //again we wait particles to be exchanged
    //then we do the same with the ghost_particles
    for (int i = 0; i < 8; i++) {
        ghost_particles.insert(ghost_particles.end(), recv_buffers[i].begin(), recv_buffers[i].end());
    }
}

//self explanitory, init for sim
void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {


    setup_2d_decomposition(num_procs); //we set up our 2d grid


    //we split the 2d sim into domains for each processor to distribute workloads
    domain_size_x = size / proc_grid_dims[0];
    domain_size_y = size / proc_grid_dims[1];
    domain_x_min = proc_grid_coords[0] * domain_size_x;
    domain_x_max = domain_x_min + domain_size_x;
    domain_y_min = proc_grid_coords[1] * domain_size_y;
    domain_y_max = domain_y_min + domain_size_y;
    
    //we make the cell size slightly larger than cutoff just in case
    cell_size = cutoff * 1.1;

    //we calc the grid size based on domain dims
    grid_size = std::max(
        static_cast<int>(domain_size_x / cell_size) + 1,
        static_cast<int>(domain_size_y / cell_size) + 1
    );
    
    //we flatten 2d to 1d
    flat_grid.resize(grid_size * grid_size);
    local_particles.reserve(num_parts / num_procs); 
    particle_indices.reserve(num_parts / num_procs * 2); 
    
    for (int i = 0; i < num_parts; i++) {
        if (is_in_domain(parts[i])) {
            particle_with_cell pwc = {parts[i], get_cell_idx(parts[i].x, parts[i].y)};
            local_particles.push_back(pwc);
        }
    }
    
    ghost_particles.clear();
    exchange_ghost_particles();
    build_grid(); 
}

//method to move particles that leave domain into new proc
void exchange_moved_particles() {
    std::vector<std::vector<particle_t>> outgoing_particles(8, std::vector<particle_t>());
    std::vector<int> indices_to_remove; 
    for (int i = 0; i < 8; i++) outgoing_particles[i].reserve(local_particles.size() / 10);

    //figure out which particles that have moved out of this domain
    for (size_t i = 0; i < local_particles.size(); i++) {
        particle_t& p = local_particles[i].p;
        if (p.x < domain_x_min) {
            if (p.y < domain_y_min && neighbors[4] != MPI_PROC_NULL) { outgoing_particles[4].push_back(p); indices_to_remove.push_back(i); }
            else if (p.y >= domain_y_max && neighbors[6] != MPI_PROC_NULL) { outgoing_particles[6].push_back(p); indices_to_remove.push_back(i); }
            else if (neighbors[0] != MPI_PROC_NULL) { outgoing_particles[0].push_back(p); indices_to_remove.push_back(i); }
        } else if (p.x >= domain_x_max) {

            if (p.y < domain_y_min && neighbors[5] != MPI_PROC_NULL) { outgoing_particles[5].push_back(p); indices_to_remove.push_back(i); }
            else if (p.y >= domain_y_max && neighbors[7] != MPI_PROC_NULL) { outgoing_particles[7].push_back(p); indices_to_remove.push_back(i); }
            else if (neighbors[1] != MPI_PROC_NULL) { outgoing_particles[1].push_back(p); indices_to_remove.push_back(i); }
        } else if (p.y < domain_y_min && neighbors[2] != MPI_PROC_NULL) { outgoing_particles[2].push_back(p); indices_to_remove.push_back(i); }
        else if (p.y >= domain_y_max && neighbors[3] != MPI_PROC_NULL) { outgoing_particles[3].push_back(p); indices_to_remove.push_back(i); }
    }
    
    MPI_Request requests[16];
    std::vector<std::vector<particle_t>> incoming_particles(8, std::vector<particle_t>());
    for (int i = 0; i < 8; i++) incoming_particles[i].reserve(local_particles.size() / 10);
    int req_count = 0;
    
    //swap particles with neighbors
    for (int i = 0; i < 8; i++) {
        int neighbor_rank = neighbors[i];
        if (neighbor_rank != MPI_PROC_NULL) {
            if (!outgoing_particles[i].empty()) {
                MPI_Isend(outgoing_particles[i].data(), outgoing_particles[i].size(), PARTICLE, neighbor_rank, 1, cart_comm, &requests[req_count++]);
            } else {
                MPI_Isend(NULL, 0, MPI_BYTE, neighbor_rank, 1, cart_comm, &requests[req_count++]);
            }
            MPI_Status status;
            MPI_Probe(neighbor_rank, 1, cart_comm, &status);
            int msg_size;
            MPI_Get_count(&status, MPI_BYTE, &msg_size);
            int num_particles = msg_size / sizeof(particle_t);
            if (num_particles > 0) {
                incoming_particles[i].resize(num_particles);
                MPI_Irecv(incoming_particles[i].data(), num_particles, PARTICLE, neighbor_rank, 1, cart_comm, &requests[req_count++]);
            } else {
                MPI_Irecv(NULL, 0, MPI_BYTE, neighbor_rank, 1, cart_comm, &requests[req_count++]);
            }
        }
    }
    
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE); //again, wait for all proc to have exchanges completed
    //clean up particles that left/were exchanged

    std::sort(indices_to_remove.begin(), indices_to_remove.end(), std::greater<int>());
    for (int idx : indices_to_remove) {
        local_particles.erase(local_particles.begin() + idx);
    }
    //add any incoming particles from swap
    for (int i = 0; i < 8; i++) {
        for (const auto& p : incoming_particles[i]) {
            if (is_in_domain(p)) {
                particle_with_cell pwc = {p, get_cell_idx(p.x, p.y)};
                local_particles.push_back(pwc);
            }
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    //reset acc
    for (auto& pwc : local_particles) {
        pwc.p.ax = 0;
        pwc.p.ay = 0;
    }
    
    //calc forces for particles
    for (size_t i = 0; i < local_particles.size(); i++) {
        particle_t& local_particle = local_particles[i].p;
        int cell_idx = local_particles[i].cell_idx;
        int cell_y = cell_idx / grid_size;          
        int cell_x = cell_idx % grid_size;          
        
        //check all grid cells 
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int nx = cell_x + dx;               
                int ny = cell_y + dy;
                
                
                //only cells within the grid
                if (nx < 0 || nx >= grid_size || ny < 0 || ny >= grid_size) continue;
                int n_idx = ny * grid_size + nx;   
                const flat_cell_t& cell = flat_grid[n_idx]; //we get the neighbor cell

                //apply the force from each particle in the neighbor cell
                for (int j = cell.start; j < cell.start + cell.count; j++) {
                    int idx = particle_indices[j];
                    if (idx < (int)local_particles.size()) { 
                        if (i != (size_t)idx) {         //dont double up on force
                            apply_force(local_particle, local_particles[idx].p);
                        }
                    } else 
                    {                           //ghost part
                        int ghost_idx = idx - local_particles.size();
                        apply_force(local_particle, ghost_particles[ghost_idx]);
                    }
                }
            }
        }
    }
    
    //we calc new pos based on new accs
    for (auto& pwc : local_particles) {
        move(pwc.p, size);
    }
    exchange_moved_particles(); //update particles
    exchange_ghost_particles(); //Update ghost particles
    build_grid();              //new grid with new pos's
}

//method to gather all particles to rank 0 for saving
void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Copy local particles
    std::vector<particle_t> temp_local(local_particles.size());
    for (size_t i = 0; i < local_particles.size(); i++) temp_local[i] = local_particles[i].p;
    int local_count = temp_local.size(); //num of particles this processor has
    int* counts = new int[num_procs];    //num of particles from each processor
    int* displs = new int[num_procs];    //offsets in the gathered data


    //get TOTAL count of particles from each processor
    MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {


        //calc offsets for where each processor’s data goes
        displs[0] = 0;
        for (int i = 1; i < num_procs; i++) {
            displs[i] = displs[i-1] + counts[i-1];
        }
    }
    //gather all particles to rank 0
    MPI_Gatherv(temp_local.data(), local_count, PARTICLE,
                parts, counts, displs, PARTICLE, 0, MPI_COMM_WORLD);
    if (rank == 0) {

        //sort by ID
        std::sort(parts, parts + num_parts, 
                  [](const particle_t& a, const particle_t& b) { return a.id < b.id; });
    }
    delete[] counts; //free up memory
    delete[] displs;
}

//method to clean up and free up memory
void finalize_simulation() {
    flat_grid.clear();       
    particle_indices.clear(); 
    local_particles.clear();  
    ghost_particles.clear();  
    MPI_Comm_free(&cart_comm); 
}
