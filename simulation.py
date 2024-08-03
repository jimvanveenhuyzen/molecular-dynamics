import numpy as np 
import matplotlib.pyplot as plt 

class Simulation:
    def __init__(self, r_init, v_init, t, dt, boxsize):

        # Pre-define some constants to use 
        self.N = np.shape(r_init)[0]
        self.time = np.arange(0,t,dt)
        self.M = len(self.time)
        self.L = boxsize 

        # Set-up the position, velocity and force arrays
        self.r = np.zeros((self.M,self.N,3))
        self.v = np.zeros((self.M,self.N,3))    
        self.r[0] = r_init
        self.v[0] = v_init
        self.f = np.zeros((self.M,self.N,3))

    def compute_force(self, r_i, r_beta):
        pass 

# Define initial positions and velocities, start with zero velocity
pos = np.random.uniform(0,3,size=(10,3))
vel = np.zeros((10,3))

dt = 5e-3
times = np.arange(0,0.5,dt)
M = len(times)

pos_total = np.zeros((M,10,3))
pos_total[0] = pos
vel_total = np.zeros((M,10,3))
#print(pos_total)

f = np.zeros((M,10,3))
L = 3
#loop through all timesteps except the last 
for t_i,elem in enumerate(times[:-1]):

    #per timestep, obtain all positions and velocities
    pos_t = pos_total[t_i]
    vel_t = vel_total[t_i]

    #print(pos_t.shape)

    #print(pos_t.shape)

    #loop through the particles 
    max = 10
    for i in range(max):

        #get current x_i
        pos_i = pos_t[i]

        if i == (max-1):
            pos_beta = pos_t[:-1]
            #print(pos_beta.shape)
        elif i == 0: 
            pos_beta = pos_t[1:]
        else: 
            pos_beta = np.concatenate((pos_t[:i],pos_t[i+1:]),axis=0)
            #print(pos_beta.shape)

        # calculate x_i - x_beta 

        diff = pos_i - pos_beta

        #print(diff.shape)

        # find |x_i - x_beta| 
        diff_norm = np.linalg.norm(diff,axis=1)

        #print(diff_norm.shape)
        
        test = diff/diff_norm[:,np.newaxis] * ((diff_norm[:,np.newaxis])**-6 - 2*(diff_norm[:,np.newaxis])**-12) #this works! 
        #print(np.sum(test,axis=0))
        #print(test.shape)

        #compute the forces due to the influence of other particles 
        for b in range(max):

            #if beta is not equal to i, execute 
            if b != i: 

                pos_beta = pos_t[b]
                #print('i',pos_i)
                #print('beta',pos_beta)
                #print(pos_beta)

                rdiff = pos_i - pos_beta 
                rdiffmag = np.linalg.norm(rdiff)
                #print(rdiffmag)

                #print(rdiff.shape)
                #print(rdiffmag.shape)
                
                f[t_i,b,:] +=  rdiff/rdiffmag * ((1/rdiffmag)**6 - 2*(1/rdiffmag)**12) 
        #print(f[t_i,i,:])

    #Now, apply Euler's method to compute the positions and velocities at the next timesteps

    pos_total[t_i+1] = pos_total[t_i] + vel_total[t_i] * elem
    vel_total[t_i+1] = vel_total[t_i] + f[t_i] * elem 

    #if pos_total[t_i+1].any() > L:
    #    print(pos_total[t_i+1])
    pos_total[t_i+1] = np.mod(pos_total[t_i+1],L)
    #pos_total[t_i+1] = np.where(pos_total[t_i+1] > L, pos_total[t_i+1] - L, pos_total[t_i+1])
    #pos_total[t_i+1] = np.where(pos_total[t_i+1] < 0, pos_total[t_i+1] + L, pos_total[t_i+1])

print('-'*100)
print(pos_total[:,0,:])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in range(M):
    #if i % 5 == 0:
        x,y,z = pos_total[i,0,0],pos_total[i,0,1],pos_total[i,0,2]
        ax.scatter(x,y,z,s=3,color='blue')

ax.set_xlim([0,L])
ax.set_ylim([0,L])
ax.set_zlim([0,L])
plt.show()