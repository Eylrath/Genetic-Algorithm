import numpy as np
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    """class being an implementation of standard genetic algorithm"""

    def __init__(self, f=None, ranges=(0,256), N=10, length=8, filename=None):
        """
        * ranges - a tuple containg ranges of given problem
        * N - number of individuals per epoch
        * max_epochs - maximal number of epochs, overstepping this value will cause stopping algorithm
        * length - length of individual
        """
        self.N = N
        if filename is None:
            self.range_min = 0 # ranges[0]
            self.range_max = ranges[1]
            self.find_length()
            self.create_domain()
            self.goal_function(f)
        else:
            self.read_from_file(filename, ";")
    
    def binary(self, X):
        """
        converts the input number array into 2D array of 0s and 1s
        in: 
            * X - input array
        out:
            * 2D array of 0s and 1s 
        """
        X_bin = []
        for x in X:
            x_b_str = bin(x)[2:]
            x_b_int = [int(n) for n in x_b_str]
            x_b_int = [0] * (self.length-len(x_b_int)) + x_b_int
            X_bin.append(x_b_int)
#         print(X_bin)
        return np.array(X_bin, dtype=np.uint32)
    
    def find_length(self):
        """
        finds the needed length for the given domain
        """
        L = self.range_max - self.range_min - 1
        length = 0
        exp = 1
        bits = []
        while L >0:
            length += 1
            L = L // 2
            bits = [exp] + bits
            exp *= 2
        self.length = length
        bits = np.array(bits)
        self.bits = np.vstack([bits] * self.N)
        
    def create_domain(self):
        """
        function creates domain by initialized earlier values 
        * P_range, range_min, range_max
        """
        X = np.arange(self.range_min, self.range_max)
        X_bin = []
        self.X = X
        self.X_bin = self.binary(X)
        
    def goal_function(self,f):
        """
        adds a certain goal function
        in:
            * f - goal function
        """
        self.f = f
        self.f_X = f(self.X)
        
    def read_from_file(self, filename, sign):
        """
        reads values from file and sets length, ranges and the rest
        in: 
            * filename - name of the file with values
            * sign - delimiter for proper reading of the file
        """
        f = open(filename, 'r')
        lines = f.readlines()
        f.close()
        X = []
        f_X = []
        for line in lines:
            tmp = line.strip().split(sign)
            X.append(np.int32(tmp[0]))
            f_X.append(np.float(tmp[1]))
        self.X = np.array(X)
        
        self.f_X = np.array(f_X)
        self.range_min = X[0]
        self.range_max = X[-1] + 1
        self.find_length()
        self.X_bin = self.binary(X)
            
          
        
    def create_population(self):
        """
        creates inital population P
        out:
            * randomly generated population of N individuals
        """
        P = np.random.choice(self.X, self.N)
        return P
    
    def evaluate_population(self, P):
        """
        returns f(individual) for every individual
        in:
            * P - population
        out:
            * array of values of goal function for the given populaiton
        """
#         self.f_P = self.f(self.P)
        return self.f_X[P]
    
    def create_ranges(self, f_P):
        """creates ranges of probability for every individual
        based on the evalution by the goal function
        in:
            * f_P - evaluated population (f(P))
        out:
            * array of growing values of probability on roulette for every individual"""
        ranges = np.cumsum(f_P) / np.sum(f_P) * 100
        return ranges
    
    def choose_parents(self, ranges, P):
        """
        randomly chooses parents for crossover
        in:
            * ranges - roulette scores in percentage for every individual
            * P - population
        out:
            * array of randomly chosen parents 
        """
        roulette_shots = np.random.randint(0,100, self.N)
        shot_indexes = np.array([np.argmin(ranges < x) for x in roulette_shots])
        np.random.shuffle(shot_indexes)
        return P[shot_indexes]
    
    def crossover(self, parents_bin, max_p=0.25):
        """
        crossover of chosen randomly individuals by the roulette rule
        in:
            * parents - individuals chosen for crossover
            * max_p - maximal probability of crossover
        out:
            * 2D array of 1s and 0s  being a new population
        """
#         parents_bin = self.X_bin[parents]
        poc = np.random.randint(0, self.length, self.N//2) #points of crossing
        children = np.zeros(self.length * self.N, dtype=np.uint32).reshape((self.N,self.length ))
        choice = np.random.uniform(0,max_p,self.N // 2)
        chance = [np.random.choice([False, True],1, p=[1-p_i, p_i])[0] for p_i in choice]
        for i in range(0, self.N, 2):
            if chance[i//2]:
                intersection = poc[i//2]
                children[i] += np.concatenate([parents_bin[i, :intersection], parents_bin[i+1,intersection:]])
                children[i+1] += np.concatenate([parents_bin[i+1, :intersection], parents_bin[i,intersection:]])
            else:
                children[i] = parents_bin[i]
                children[i+1] = parents_bin[i+1]
        return children
    
    def mutation(self, P_bin, max_p=0.1):
        """
        procedure allowing some individuals to mutate according to random probability
        in:
            * P_bin - population in binary representation
            * max_p - maximal probability of mutation
        out:
            * new population with (perhaps) different memebers
        """
        choice = np.random.uniform(0,max_p,self.N )
        chance = [np.random.choice([False, True],1, p=[1-p_i, p_i])[0] for p_i in choice]
        for i in range(self.N):
            if chance[i]:
                idx = np.random.randint(0, self.length - 1)
                P_bin[i,idx] = (P_bin[i,idx]+1)%2
        return P_bin
    
    def binary_to_int(self, P_bin):
        """
        converts 2D array of actual population into values represented by it
        in:
            * P_bin - population in binary representation
        out:
            * representation in integer numbers
        """
        
#         children = np.sum(self.bits[np.bool_(P_bin), :],axis=1)
        z = np.ma.array(self.bits, mask=~np.bool_(P_bin), fill_value=0)
        children = np.sum(z, axis=1).compressed()
        return children
    
    def int_to_binary(self, P):
        """returns binary representation of array
        in:
            * P - population
        out:
            * array of binary representations of all individuals from P
        """
        return self.X_bin[P]
    
    def mean(self, P):
        """
        retuns mean of actual generation
        in:
            * P - population
        out:
            * mean of P
        """
        return np.mean(P)
    
    def max(self, P):
        """
        returns the max value of actual generation
        in:
            * P - population
        out:
            * maximal-value individual of the generation
        """
        return P[np.argmax(self.f_X[P])]
    
    def child_parent_changes(self, children, parents):
        """
        checks difference between actual generation and the upcoming one
        in:
            * children - newpopulation
            * parents - old population
        out:
            * percentage of repeating individuals 
        """
        return len(np.intersect1d(children,parents)) / len(children)
    
    def plot_ranges(self, population):
        """
        plots ranges of a generation
        """
        P, counts = np.unique(population, return_counts=True)
        fp = self.f_X[P] * counts
        plt.pie(fp, labels=P)
        plt.show()
        
    def elitism(self, P, new_P):
        """
        makes sure, that the best individual from previous generation lasts
        in:
            * P - population
            * new_P - new population
        out:
            * population with the best individual from the previous generation on random position
        """
        max_ind = self.max(P)
        new_P[np.random.randint(len(new_P))] = max_ind
        new_P[0] = max_ind
        return new_P
    