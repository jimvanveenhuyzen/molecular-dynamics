import numpy as np 
import matplotlib.pyplot as plt 
import timeit

# Define initial positions and velocities, start with zero velocity
pos = np.random.uniform(0,3,size=(10,3))
vel = np.zeros((10,3))

dt = 0.032
times = np.arange(0,1,dt)
M = len(times)

pos_total = np.zeros((M,10,3))
pos_total[0] = pos
vel_total = np.ones((M,10,3))*0.1
#print(pos_total)

f = np.zeros((M,10,3))
L = 3
#loop through all timesteps except the last

begin = timeit.default_timer()

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
            pos_beta1 = pos_t[:-1]
            #print(pos_beta.shape)
        elif i == 0: 
            pos_beta1 = pos_t[1:]
        else: 
            pos_beta1 = np.concatenate((pos_t[:i],pos_t[i+1:]),axis=0)
            #print(pos_beta.shape)

        # calculate x_i - x_beta 

        diff = pos_i - pos_beta1

        #print(diff.shape)

        # find |x_i - x_beta| 
        diff_norm = np.linalg.norm(diff,axis=1)

        #print(diff_norm.shape)
        
        #first try:
        #test = diff/diff_norm[:,np.newaxis] * ((diff_norm[:,np.newaxis])**-6 - 2*(diff_norm[:,np.newaxis])**-12) #this works! 

        #second try (22-7):
        test = diff * ((diff_norm[:,np.newaxis])**(-14) - 0.5*(diff_norm[:,np.newaxis])**(-8))
        test_sum = np.sum(test,axis=0)
        #print(np.sum(test,axis=0))
        #print(test_sum.shape)
        f[t_i,i] = test_sum

        #compute the forces due to the influence of other particles 
        # for b in range(max):

        #     #if beta is not equal to i, execute 
        #     if b != i: 

        #         pos_beta = pos_t[b]
        #         #print('i',pos_i)
        #         #print('beta',pos_beta)
        #         #print(pos_beta)

        #         rdiff = pos_i - pos_beta 
        #         rdiffmag = np.linalg.norm(rdiff)
        #         #print(rdiffmag)

        #         #print(rdiff.shape)
        #         #print(rdiffmag.shape)
                
                #f[t_i,b,:] +=  rdiff/rdiffmag * ((1/rdiffmag)**6 - 2*(1/rdiffmag)**12) 
        #print(f[t_i,i,:])

    #Now, apply Euler's method to compute the positions and velocities at the next timesteps

    pos_total[t_i+1] = pos_total[t_i] + vel_total[t_i] * elem
    vel_total[t_i+1] = vel_total[t_i] + f[t_i] * elem 
    #pos_total[t_i+1] = pos_total[t_i] + vel_total[t_i] * elem

    #if pos_total[t_i+1].any() > L:
    #    print(pos_total[t_i+1])
    pos_total[t_i+1] = np.mod(pos_total[t_i+1],L)

    #pos_total[t_i+1] = np.where(pos_total[t_i+1] > L, pos_total[t_i+1] - L, pos_total[t_i+1])
    #pos_total[t_i+1] = np.where(pos_total[t_i+1] < 0, pos_total[t_i+1] + L, pos_total[t_i+1])

end = timeit.default_timer()-begin

print('-'*100)
print(pos_total[:,0,:])
print(vel_total[:,0,:])
print(f[:,0,:])

# for i in range(M):
#     plt.scatter(pos_total[i,0,0],pos_total[i,0,1],color='blue')
#     plt.scatter(pos_total[i,1,0],pos_total[i,1,1],color='black')

# plt.xlim([0,L])
# plt.ylim([0,L])
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in range(M):
    #if i % 5 == 0:
        x,y,z = pos_total[i,0,0],pos_total[i,0,1],pos_total[i,0,2]
        x2,y2,z2 = pos_total[i,1,0],pos_total[i,1,1],pos_total[i,1,2]
        ax.scatter(x,y,z,color='blue')
        ax.scatter(x2,y2,z2,color='black')

ax.set_xlim([0,L])
ax.set_ylim([0,L])
ax.set_zlim([0,L])
plt.show()

print('This took {:.4f} seconds'.format(end))