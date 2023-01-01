import itertools as itr
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm 
import pandas as pd
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d


MIU_SIZE = 14
ITER_COUNT = 25 # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
EPS = 0.003
U_BOUND = 4
L_BOUND = -2
DIM = 30

class hypervol_solver():
    
    def __init__(self, funcs, given_x=np.array([])):
        self.counter = 0
        
        if len(funcs) > 3 or len(funcs) < 1 : 
            raise Exception('1-3 objective functions per solver')
        self.funcs = funcs
        
        if given_x.size:
            self.x_vec = given_x

        else:
            self.x_vec = np.random.uniform(L_BOUND, U_BOUND, (500, DIM))

        self.y_vec = self.evaluation(self.x_vec) 
        
    def evaluation (self, x_vec):
        y_vec = np.array([[self.funcs[i](x_vec[j]) for j in range \
             (x_vec.shape[0])] for i in range (len(self.funcs))] ).T
       
        return y_vec


    def solve(self):
        self.build_front()
   
        diff = np.inf
        self.vols = []
        vol = 0
        if self.front_size > MIU_SIZE:
            vol = self.hypervol() #0 will be counted twice in this case
            self.vols.append(vol) 
        while(self.counter < ITER_COUNT): # (diff > EPS) and  #stop conditions TODO change 
            print("ITERCOUNT: ", self.counter)
            self.counter += 1
            self.iterate()
            vol_n = self.hypervol() #remove least contributors
            self.vols.append(vol_n)
            diff = vol_n - vol  #add new point, must remain undominated set
            
            vol = vol_n 
        return self.front
    
    
    def build_front(self):
        y = self.is_pareto_efficient_simple(self.y_vec)
        self.front_size = sum(y)
        miu_y = self.y_vec[y]
        miu_x = self.x_vec[y] 

        self.front = [np.zeros((miu_x.shape)), np.zeros((miu_y.shape))]
        self.front[0] = np.copy(miu_x)
        self.front[1] = np.copy(miu_y)
        

    def is_pareto_efficient_simple(self, costs):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype = bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        return is_efficient
        
    def iterate(self): 
        #while self.front[0].shape[0] < 100 :
        for pair in list (itr.combinations(self.front[0], 2)) : #to prevent population extinction 
            self.front[0] = np.append(self.front[0], self.recombine(pair[0],pair[1]).reshape(1,DIM), axis=0)

        self.y_vec = self.evaluation(self.front[0])
        self.x_vec = self.front[0]
        print('pop before is: ' + str(self.front[0].shape[0]))
        self.build_front()
        print('pop after is: ' + str(self.front[0].shape[0]))

    def recombine(self, p1,p2): 
        offset = np.random.randint(1,p1.shape[0])
        mut = np.random.randint(0,p1.shape[0]) #maybe make this less now!

        new_p = np.concatenate((p1[:offset],p2[offset:]),axis=0)
        new_p[mut] = np.random.normal(np.mean(p1), 0.2)
        return new_p



    def remove_edges(self, pop):
        dim = pop.shape[1]
        indices = np.zeros(pop.shape[1])
        for k in range(dim):
            indices[k] = int(pop[:,k].argmax())
        indices = np.unique(indices).astype(int)
        edg = pop[indices]
        pop = np.delete(pop, indices, axis=0)
        return pop, edg
        
    def compute_hypervol(self, given_ps, ref_p, i, vol, scan_z, final_vol):

        if i == 0: 
            final_vol += vol
            return final_vol
        
        elif i > 0:
            front_by_i = np.copy(given_ps) 
            dim = len(scan_z)
            up = np.copy(front_by_i) #initiated also if doesnt enter
            for j in range(i,dim): 
                up = up[up[:,j] <= scan_z[j]]

            U = np.concatenate((up, ref_p), axis=0) 

            while U.size != 0 :
                u_star = U[ U[:,i-1].argmin() ] [i-1] # 
                U_tag = U[ U[:,i-1] > u_star ]
                if U_tag.size != 0 :
                    vol_tag = vol * (U_tag[U_tag[:,i-1].argmin()][i-1] - u_star)
                    scan_send = np.copy(scan_z)
                    scan_send[i-1] = u_star
                    final_vol = self.compute_hypervol(front_by_i, ref_p, i-1, vol_tag, scan_send, final_vol)
                U = U_tag 
        return final_vol 
    
    def init_dystopia(self) : #any dimensional
        miu = np.copy(self.front[1])
        dim = len(self.funcs) # miu[0].shape[0]  #WHY NOT JUST 

        dystopia = np.full(dim,np.inf)
        for m in range(dim) :
            dystopia[m] = np.max(miu[:,m]) #CHANGED MIU TO Y_VEC #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        return dystopia #DO I NEED TO MAKE A COPY OF Y_VEC?
    
    def hypervol (self):
        reference_point = self.init_dystopia()  #ref point is the worse x,y,(z) of all miu 
        reference_point = reference_point.reshape(1, reference_point.shape[0])
        
        dim = reference_point.shape[1]#volume to be filled 
        zn = [np.inf] * dim
        
        without_edges, edgs = self.remove_edges(self.front[1])
        print("original size of front is: " + str(self.front[1].shape[0]))
    
        if dim == 3 :            
            while self.front_size > MIU_SIZE:
                max_vol = 0
                min_ind = 0
                for i in range(without_edges.shape[0]):
                    if i%10==0 and self.front_size%5==0 :
                        print(i)
                    t = np.delete(without_edges, i, axis=0)
                    v = self.compute_hypervol(t, reference_point, 3, 1, zn, 0)
                    if v > max_vol:
                        max_vol = v
                        min_ind = i

                loser = without_edges[min_ind]
                i_remove = np.all(self.front[1]==[loser], axis=1)
                i_keep = i_remove == False
                self.front[1] = self.front[1][i_keep]
                self.front[0] = self.front[0][i_keep]
                
                without_edges = np.delete(without_edges, min_ind, axis=0)
                
                self.front_size -= 1
            
            #with_edges = np.append(without_edges, edgs, axis=0)
            
            fin_vol = self.compute_hypervol(without_edges, reference_point, 3, 1, zn, 0)
            #def compute_hypervol(self, given_ps, ref_p, dim, vol, scan_z):
            print("final vol is: ", fin_vol)
            
           
           
        fig = plt.figure()#figsize = (36, 27))
        ax1 = fig.add_subplot(111, projection='3d')  
    
        ax1.plot_surface(np.array([self.front[1][:,0], self.front[1][:,1]]), np.array([self.front[1][:,0], self.front[1][:,2]]), \
                         np.array([self.front[1][:,1], self.front[1][:,2]]), cmap=cm.coolwarm)
    #ax1.scatter(self.front[1][:,0], self.front[1][:,1], self.front[1][:,2], color = 'green')
        ax1.scatter(reference_point[0,0], reference_point[0,1], reference_point[0,2], color = 'red')
        titl = "Iteration: " + str(self.counter)
        plt.title(titl)
        #plt.pause(1)
        plt.show()
        
        
        combinations = [np.array([self.front[1][:,0], self.front[1][:,1]]), \
                        np.array([self.front[1][:,0], self.front[1][:,2]]), \
                            np.array([self.front[1][:,1], self.front[1][:,2]])]
        xy = combinations[0]
        xz = combinations[1]
        yz = combinations[2]
                
        fig = plt.figure()
        ax2 = plt.scatter(xy[0], xy[1])
        plt.title("XY axis" + " Iteration: " + str(self.counter))
        plt.xlabel('x  f(x)=(x)^2')
        plt.ylabel('y  f(x)=(x-1)^2')

        fig = plt.figure()
        ax3 = plt.scatter(xz[0], xz[1])
        plt.title("XZ axis" + " Iteration: " + str(self.counter))
        plt.xlabel('x  f(x)=(x)^2')
        plt.ylabel('z  f(x)=(x-2)^2')
        
        fig = plt.figure()
        ax3 = plt.scatter(yz[0], yz[1])
        plt.title("YZ axis" + " Iteration: " + str(self.counter))
        plt.xlabel('y  f(x)=(x-1)^2')
        plt.ylabel('z  f(x)=(x-2)^2')
        
        plt.show()

        return fin_vol


    def best_point(self):
        y = self.front[1][np.argmin(np.linalg.norm(self.front[1], axis=1))]
        x = self.front[0][np.argmin(np.linalg.norm(self.front[1], axis=1))]
        p = np.min(np.linalg.norm(self.front[1], axis=1))
        return p,y,x

# solve for n = 30
f3_1 = lambda x1 : np.dot(x1.T , x1) # minimize
f3_2 = lambda x1 : np.dot((x1-1).T, x1-1) # minimize
f3_3 = lambda x1 : np.dot((x1-2).T, x1-2) # minimize
funcs = [f3_1, f3_2, f3_3]


first_front_x_list = [[0.73840038, 0.83569956, 0.58592008, 0.45580946, 0.73127621,
         0.54853934, 0.47288092, 0.65775676, 0.60282831, 0.38093851,
         0.29340952, 0.25206192, 0.40821473, 0.33096667, 0.69938069,
         0.7753754 , 0.15260016, 0.26247777, 0.32831247, 0.42267997,
         0.68263977, 0.86215964, 0.52075528, 0.35765666, 0.52270024,
         0.60384729, 0.6388486 , 0.78542322, 0.41624243, 0.44355765],
        [2.20598132, 2.43443422, 1.79167975, 1.73518301, 1.54151773,
         1.53452306, 1.4427594 , 1.42981977, 1.50379941, 1.51450879,
         1.64343249, 2.0075628 , 1.11337254, 1.75262162, 1.45045932,
         1.39167276, 1.6928765 , 1.13315894, 1.83841234, 1.96962682,
         1.56275538, 1.50878228, 1.10238113, 1.5948163 , 1.35099194,
         1.58424773, 1.27812068, 1.36482406, 1.80998076, 1.45456906],
        [1.15213341, 0.90960856, 1.16189286, 0.86562341, 1.00924527,
         1.13531585, 0.95720903, 1.03783263, 1.03121024, 1.04864316,
         1.03756646, 0.92818644, 1.11337254, 1.20846458, 1.00637802,
         1.05147156, 1.09842768, 1.03094029, 0.91242708, 1.0497931 ,
         0.96792834, 0.94325251, 0.95391995, 0.94055349, 0.98130018,
         0.98004115, 0.9075965 , 0.99700059, 0.93936164, 0.9436119 ],
        [0.73840038, 0.83569956, 0.58592008, 0.45580946, 0.73127621,
         0.54853934, 0.47288092, 0.65460999, 0.60282831, 0.7252522 ,
         0.63395576, 0.45687496, 0.77726579, 0.64393076, 0.69938069,
         0.77137026, 0.60947352, 0.61463556, 0.32831247, 0.42267997,
         0.68263977, 0.86215964, 0.52075528, 0.35765666, 0.52270024,
         0.74381605, 0.6388486 , 0.64541732, 0.55286672, 0.65560108],
        [0.84054237, 0.83569956, 0.71845881, 0.86562341, 0.73127621,
         1.05609681, 0.91638443, 0.81132582, 1.03121024, 0.7252522 ,
         1.03756646, 0.84172292, 0.77726579, 0.78430256, 0.69938069,
         0.92686322, 0.91328672, 0.92112092, 0.66536595, 0.87164318,
         0.96322399, 0.94325251, 0.95391995, 0.94055349, 0.98130018,
         0.98004115, 0.9075965 , 0.99700059, 0.97657089, 0.9436119 ],
        [0.84054237, 0.83569956, 0.71845881, 0.86562341, 0.73127621,
         1.05609681, 0.91638443, 0.81132582, 1.03121024, 0.7252522 ,
         1.03756646, 0.92818644, 0.77726579, 1.20085523, 0.69938069,
         0.92686322, 0.91328672, 0.92112092, 0.91242708, 1.0497931 ,
         0.96792834, 0.94325251, 0.95391995, 0.94055349, 0.98130018,
         0.98004115, 0.9075965 , 0.99700059, 0.97657089, 0.9436119 ],
        [1.15213341, 1.33621021, 1.281098  , 1.41647891, 1.44842197,
         1.30529254, 1.4427594 , 1.38276554, 1.23044364, 1.07990776,
         1.38580136, 1.15179807, 1.32817109, 1.26872204, 1.17645934,
         1.22438616, 1.21218192, 1.22713057, 1.41901516, 1.3357692 ,
         1.38084362, 1.50878228, 1.40024642, 1.13501353, 1.22410592,
         0.98004115, 1.21140258, 1.16125281, 1.15210271, 1.17745532],
        [1.15213341, 0.90960856, 1.16189286, 0.86562341, 1.00924527,
         1.13531585, 0.95720903, 1.03783263, 1.03121024, 1.04864316,
         1.03756646, 1.15179807, 1.10541301, 1.2468986 , 1.17645934,
         1.22438616, 1.21218192, 1.21453991, 1.06515541, 1.21561337,
         1.06102361, 1.16457632, 1.25139097, 1.13501353, 1.13575728,
         0.98004115, 1.21140258, 1.16125281, 1.15210271, 1.17745532],
        [0.73840038, 0.83569956, 0.58592008, 0.45580946, 0.73127621,
         0.54853934, 0.47288092, 0.65460999, 0.60282831, 0.7252522 ,
         0.58371937, 0.45687496, 0.77726579, 0.64393076, 0.69938069,
         0.77137026, 0.70233547, 0.88854714, 0.66536595, 0.61895761,
         0.96322399, 0.80561605, 0.57128283, 0.9491119 , 0.54828688,
         0.74381605, 0.6388486 , 0.64541732, 0.55286672, 0.65560108],
        [0.73840038, 0.83569956, 0.58592008, 0.45580946, 0.73127621,
         0.54853934, 0.47288092, 0.65460999, 0.60282831, 0.7252522 ,
         0.58371937, 0.45687496, 0.77726579, 0.64393076, 0.69938069,
         0.58059092, 0.70233547, 0.88854714, 0.66536595, 0.61895761,
         0.96322399, 0.80561605, 0.93762621, 0.9491119 , 0.91353874,
         0.74381605, 0.68035542, 0.64541732, 0.87921843, 0.74948965],
        [0.84054237, 0.64342373, 0.71845881, 0.86562341, 0.73127621,
         0.78510617, 0.95720903, 0.81132582, 1.03121024, 0.7252522 ,
         0.63395576, 0.92818644, 0.77726579, 0.64393076, 0.69938069,
         0.77137026, 0.70233547, 0.88854714, 0.66536595, 0.61895761,
         0.96322399, 0.80561605, 0.93762621, 0.9491119 , 0.54828688,
         0.74381605, 0.6388486 , 0.64541732, 0.55286672, 0.74948965],
        [0.84054237, 0.83569956, 0.71845881, 1.13333467, 0.73127621,
         1.05609681, 0.91638443, 0.81132582, 1.03121024, 1.04864316,
         1.03756646, 0.92818644, 1.11337254, 1.20085523, 0.69938069,
         0.92686322, 0.91328672, 0.92112092, 0.91242708, 1.0497931 ,
         0.96792834, 0.94325251, 0.95391995, 0.94055349, 0.98130018,
         0.98004115, 0.9075965 , 0.99700059, 0.97657089, 0.9436119 ],
        [0.7374972 , 0.83569956, 0.58592008, 0.86562341, 0.73127621,
         0.78510617, 0.95720903, 0.81132582, 1.03121024, 1.012083  ,
         0.63395576, 0.92818644, 0.77726579, 0.64393076, 0.69938069,
         0.92686322, 0.91328672, 0.92112092, 0.77716953, 0.87164318,
         0.96322399, 0.80561605, 0.93762621, 0.9491119 , 0.91353874,
         0.74381605, 0.68035542, 0.64541732, 0.87921843, 0.74948965],
        [1.48481263, 1.64056756, 1.281098  , 1.41647891, 1.54151773,
         1.84970004, 1.4427594 , 1.42981977, 1.53323947, 1.56869927,
         1.64343249, 2.0075628 , 1.11337254, 1.75262162, 1.60276359,
         1.22438616, 1.3952403 , 1.22713057, 1.52971042, 1.3357692 ,
         1.38084362, 1.50878228, 1.40024642, 1.5948163 , 1.35099194,
         1.58424773, 1.27812068, 1.3530781 , 1.66141755, 1.17745532],
        [1.48481263, 1.41478238, 1.36098127, 1.41647891, 1.44842197,
         1.30529254, 1.4427594 , 1.38276554, 1.50379941, 1.58590108,
         1.38580136, 1.15179807, 1.45751795, 1.2468986 , 1.60276359,
         1.45260024, 1.3952403 , 1.22713057, 1.41901516, 1.3357692 ,
         1.38084362, 1.16457632, 1.25139097, 1.13501353, 1.13575728,
         1.25143143, 1.21140258, 1.16125281, 1.15210271, 1.17745532],
        [1.48481263, 1.41478238, 1.36098127, 1.41647891, 1.44842197,
         1.30529254, 1.4427594 , 1.38276554, 1.50379941, 1.50830886,
         1.38580136, 1.15179807, 1.45751795, 1.2468986 , 1.60276359,
         1.45260024, 1.3952403 , 1.22713057, 1.41901516, 1.3357692 ,
         1.38084362, 1.50878228, 1.40024642, 1.5948163 , 1.35099194,
         1.58424773, 1.21140258, 1.16125281, 1.15210271, 1.17745532],
        [1.48481263, 1.41478238, 1.36098127, 1.41647891, 1.44842197,
         1.30529254, 1.4427594 , 1.38276554, 1.50379941, 1.58590108,
         1.38580136, 1.15179807, 1.45751795, 1.2468986 , 1.60276359,
         1.45260024, 1.3952403 , 1.22713057, 1.41901516, 1.43139424,
         1.38084362, 1.50878228, 1.40024642, 1.5948163 , 1.35099194,
         1.58424773, 1.27812068, 1.3530781 , 1.66141755, 1.17745532],
        [1.15213341, 0.90960856, 1.16189286, 0.86562341, 1.00924527,
         1.13531585, 0.95720903, 1.03783263, 1.03121024, 1.04864316,
         1.03756646, 0.92818644, 1.11337254, 1.20846458, 1.00637802,
         1.08858723, 1.09842768, 1.03094029, 0.91242708, 1.11858015,
         0.96355822, 1.16457632, 1.20954203, 1.13501353, 1.22410592,
         0.98004115, 1.21140258, 1.16125281, 1.15210271, 1.17745532],
        [1.15213341, 1.33621021, 1.281098  , 1.41647891, 1.44842197,
         1.30529254, 1.4427594 , 1.38276554, 1.23044364, 1.07990776,
         1.22756993, 1.30444177, 1.10541301, 1.2468986 , 1.17645934,
         1.22438616, 1.21218192, 1.21453991, 1.06515541, 1.21561337,
         0.96792834, 1.16457632, 1.25139097, 1.13501353, 1.13575728,
         0.98004115, 1.21140258, 1.16125281, 1.15210271, 1.17745532],
        [1.15213341, 1.33621021, 1.281098  , 0.95182012, 1.23160644,
         1.30529254, 1.0999312 , 1.03783263, 1.03121024, 1.04864316,
         1.03007736, 1.15179807, 1.10541301, 1.2468986 , 1.17645934,
         1.22438616, 1.39427091, 1.21453991, 1.13010935, 1.21561337,
         1.17918368, 1.16457632, 1.25139097, 1.13501353, 1.13575728,
         0.98004115, 1.21140258, 1.16125281, 1.15210271, 1.17745532]]

first_front_x = np.array(first_front_x_list)

maxpts = np.zeros((3,30))
maxpts[1]=1 
maxpts[2]=2

upgraded_x = np.append(first_front_x, maxpts, axis=0)


# solver = hypervol_solver(funcs)             # MIU = 20 , ITER = 50 , given_x = 0
# solver = hypervol_solver(funcs, upgraded_x) # MIU = 14 , ITER = 25, given_x = upgraded_x 
#solver = hypervol_solver(funcs)               # MIU = 13 , ITER = 25, given_x = 0 
solver = hypervol_solver(funcs)               # MIU = 13 , ITER = 50, given_x = 0 

front = solver.solve()

print(front[1][np.argmin(np.linalg.norm(front[1], axis=1))])
# print(front)
x = front[0][np.argmin(np.linalg.norm(front[1], axis=1))]#
