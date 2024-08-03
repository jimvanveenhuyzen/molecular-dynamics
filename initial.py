import numpy as np
import matplotlib.pyplot as plt 

# Pre-define some constants
k_B = 8.617e-5 #eV/K
eps = 119.8 * k_B #eV
sigma = 3.405 #Angstrom 
m = 3.73e10 #mass in eV

m = 1

class Particle:
    def __init__(self,r,v):
        # Position
        self.r = r
         # Velocity
        self.v = v

    def distance(self, r_beta):
        self.d = self.r - r_beta
        return self.d/np.linalg.norm(self.d)
    
    def force(self, other_particles):
        # other_particles array with all particle positions except current particle 
        N = len(other_particles)
        self.force = 0
        for i in range(N):
            rbeta = other_particles[i]
            r_diff = self.r - rbeta 
            r_diff_mag = np.lingalg.norm(r_diff)
            self.force += (-24 * eps / r_diff_mag) * ((sigma/r_diff_mag)**6 - 2*(sigma/r_diff_mag)**12) * r_diff/r_diff_mag
        return self.force
    
    def compute_magnitude(self, r):
        self.mag = np.sqrt( r[0]**2 + r[1]**2 + r[2]**2)
        return self.mag

    def compute_distance(self, r1, r2):
        # Input two arrays of size (1,3) r1, r2
        dist = r1 - r2 
        return dist/self.mag
    
    def compute_force(self, particle_positions, r_i):
        # Compute the force between particles
        N = len(particle_positions)
        force = 0
        for i in range(N):
            r_beta = particle_positions[i]
            if i > 0:
                force += (-24 * eps / Particle.compute_magnitude(self,r_i)) * Particle.compute_distance(self,r_i,r_beta)

        return force 

# Define initial positions and velocities, start with zero velocity
pos = np.random.uniform(-3,3,size=(10,3))
vel = np.zeros((10,3))

dt = 1e-2
times = np.arange(0,1,dt)
M = len(times)

pos_total = np.zeros((M,10,3))
pos_total[0] = pos
vel_total = np.zeros((M,10,3))
#print(pos_total)

f = np.zeros((M,10,3))
L = 10
#loop through all timesteps
for idx,elem in enumerate(times):

    #per timestep, obtain all positions and velocities
    pos_t = pos_total[idx]
    vel_t = vel_total[idx]

    #print(pos_t.shape)

    #print(pos_t.shape)

    #loop through the particles 
    for i in range(10):

        #get current x_i and v_i 
        pos_i = pos_t[i]
        vel_i = vel_t[i]



        #compute the forces due to the influence of other particles 
        for b in range(10):

            #if beta is not equal to i, execute 
            if b != i: 

                pos_beta = pos_t[b]
                vel_beta = vel_t[b]
                #print('i',pos_i)
                #print('beta',pos_beta)
                #print(pos_beta)

                rdiff = pos_i - pos_beta 
                rdiffmag = np.linalg.norm(rdiff)
                #print(rdiffmag)

                #print(rdiff.shape)
                #print(rdiffmag.shape)
                
                f[idx,b,:] +=  rdiff/rdiffmag * ((1/rdiffmag)**6 - 2*(1/rdiffmag)**12) 

    #Now, apply Euler's method to compute the positions and velocities at the next timesteps

    if idx < (len(times)-1): # extra condition to not exceed the max time 
        pos_total[idx+1] = pos_total[idx] + vel_total[idx] * elem
        vel_total[idx+1] = vel_total[idx] + f[idx] * elem 

        pos_total[idx+1] = np.where(pos_total[idx+1] > L, pos_total[idx+1] - L, pos_total[idx+1])
        pos_total[idx+1] = np.where(pos_total[idx+1] < -1*L, pos_total[idx+1] + L, pos_total[idx+1])


print(pos_total)
print(pos_total[:,0,:])
print(vel_total[:,0])

"""
for i in range(0,500):
    x,y = pos_total[i,0,0],pos_total[i,0,1]
    plt.scatter(x,y,color='black',s=1)

plt.xlim([-10,10])
plt.ylim([-10,10])
plt.show()
"""

positions = np.array([[0,0.5,4],[1,2,6]])
#print(f[0])

part1 = Particle(positions[0],[0,0,0])
part2 = Particle(positions[1],[0,0,0])

print(part1.compute_magnitude(part1.r))

print(part1.compute_distance(part1.r,part2.r))

print(part1.compute_force(positions,part1.r))

def euler(particle,x_init,v_init,timesteps):

    positions = np.zeros((len(timesteps),3))
    velocities = np.zeros((len(timesteps),3))

    positions[0] = x_init
    velocities[0] = v_init

    for t_i,h in enumerate(timesteps):
        positions[t_i+1] = positions[t_i] + velocities[t_i] * h
        velocities[t_i+1] = velocities[t_i] + 1/m * h #* particle.

    #x_next = x + v*h
    #v_next = v + 1/m * force * h

# x = np.array([1,0,1])
# v = np.array([-2,-1,3])
# for h in time:
#     x = x + v * h
#     v = v + h
#     plt.scatter(x[0],x[1])
#     print(r'At time {:.2f}, the particle has position {} and velocity {}'.format(h,x,v))










