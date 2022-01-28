import numpy as np
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    """
    class being an implementation of standard genetic algorithm
    """
    def __init__(self, f=None, ranges=(0,256), N=10, precision=10, filename=None):
        """
        initializator
        in:
            * f - goal function
            * ranges - values to determine the domain
            * N - number of individuals per epoch
            * filename - a file containing data to read
        """
        self.N = N
#         if f is None and filename is None:
#             raise("No data given!")
        if filename is None:
            self.range_min = ranges[0] # ranges[0]
            self.range_max = ranges[1]
#             self.find_length()
            self.length = precision
            self.amount_of_points = 2**precision
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
        bin_domain = list(range(self.range_min, self.range_max))
        X_bin = []
        for x in bin_domain:
            x_b_str = bin(x)[2:]
            x_b_int = [int(n) for n in x_b_str]
            x_b_int = [0] * (self.length-len(x_b_int)) + x_b_int
            X_bin.append(x_b_int)
            
        self.bin_domain = np.array(bin_domain)
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
        X = np.linspace(self.range_min, self.range_max, self.amount_of_points)
        X_bin = []
        bin_domain = list(range(0, self.amount_of_points))
        for x in bin_domain:
            x_b_str = bin(x)[2:]
            x_b_int = [int(n) for n in x_b_str]
            x_b_int = [0] * (self.length-len(x_b_int)) + x_b_int
            X_bin.append(x_b_int)
        self.bin_domain = np.array(bin_domain)
        self.X = X
        self.X_bin = np.array(X_bin, dtype=np.uint32)
#         bits = np.array(self.length)
#         self.bits = np.vstack([bits] * self.N)
        self.bits = np.array([2 ** np.arange(self.length, dtype=np.uint32)]*self.N,dtype=np.uint32)
        
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
            f_X.append(float(tmp[1]))
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
        P = np.random.choice(self.bin_domain, self.N)
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
        """
        creates ranges of probability for every individual
        based on the evalution by the goal function
        in:
            * f_P - evaluated population (f(P))
        out:
            * array of growing values of probability on roulette for every individual
        """
        min_ = np.min(f_P)
        f_P = f_P + min_
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
        z = np.ma.array(self.bits, mask=~np.bool_(P_bin), fill_value=0)
        children = np.sum(z, axis=1).compressed()
        return children
    
    def int_to_binary(self, P):
        """
        returns binary representation of array
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
        return np.mean(self.f_X[P])
    
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
        
    def plot_data(self):
        """
        plots the whole given or generated data
        """
        plt.plot(self.X, self.f_X)
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
    



def run_GA_max_epochs(N, precision, ranges, f,max_epochs=1000, p_c=0.8, p_m=0.1):
    GA = GeneticAlgorithm(N=N, f=f, ranges=ranges, precision=precision)
    # A = GeneticAlgorithm(N=40, filename="123.txt")
    # GeneticAlgorithm
    maxes = []
    means = []

    rate = 0
    idx = 0
    population = GA.create_population()
#     GA.plot_data()
    # print(population)
    while idx < max_epochs:
        idx += 1
        maxes.append(GA.max(population))
        means.append(GA.mean(population))
        f_population = GA.evaluate_population(population)
        ranges = GA.create_ranges(f_population)
        binary_population = GA.int_to_binary(population)
        parents = GA.choose_parents(ranges, binary_population)

        children = GA.crossover(parents, p_c)
        mutation = GA.mutation(children, p_m)

        new_population = GA.binary_to_int(mutation)
        # new_population = GA.elitism(population, new_population)
        rate = GA.child_parent_changes(population, new_population)
        population = new_population
    return population, GA.X, GA.f_X, GA.max(population), idx
    # population = A.generation_change(mutation)

def run_GA_max_epochs_elitism(N, precision, ranges, f,max_epochs=1000, p_c=0.8, p_m=0.1):
    GA = GeneticAlgorithm(N=N, f=f, ranges=ranges, precision=precision)
    # A = GeneticAlgorithm(N=40, filename="123.txt")
    # GeneticAlgorithm
    maxes = []
    means = []

    rate = 0
    idx = 0
    population = GA.create_population()
#     GA.plot_data()
    # print(population)
    while idx < max_epochs:
        idx += 1
        maxes.append(GA.max(population))
        means.append(GA.mean(population))
        f_population = GA.evaluate_population(population)
        ranges = GA.create_ranges(f_population)
        binary_population = GA.int_to_binary(population)
        parents = GA.choose_parents(ranges, binary_population)

        children = GA.crossover(parents, p_c)
        mutation = GA.mutation(children, p_m)

        new_population = GA.binary_to_int(mutation)
        new_population = GA.elitism(population, new_population)
        rate = GA.child_parent_changes(population, new_population)
        population = new_population
    return population, GA.X, GA.f_X, GA.max(population), idx

def run_GA_stop_condition_1(N, precision, ranges, f,max_epochs=1000, p_c=0.8, p_m=0.1):
    GA = GeneticAlgorithm(N=N, f=f, ranges=ranges, precision=precision)
    # A = GeneticAlgorithm(N=40, filename="123.txt")
    # GeneticAlgorithm
    maxes = []
    means = []

    rate = 0
    idx = 0
    population = GA.create_population()
#     GA.plot_data()
    # print(population)
    stop_idx = max_epochs
    count_idx = 0
    while idx < max_epochs:
        idx += 1
        maxes.append(GA.max(population))
        means.append(GA.mean(population))
        f_population = GA.evaluate_population(population)
        ranges = GA.create_ranges(f_population)
        binary_population = GA.int_to_binary(population)
        parents = GA.choose_parents(ranges, binary_population)

        children = GA.crossover(parents, 0.9)
        mutation = GA.mutation(children, 0.1)

        new_population = GA.binary_to_int(mutation)
        # new_population = GA.elitism(population, new_population)
        rate = GA.child_parent_changes(population, new_population)
        if GA.max(population) == GA.max(new_population):
            if stop_idx == max_epochs:
                stop_idx = int((idx+20) * 0.9)
            count_idx += 1
            if count_idx == stop_idx:
                break
        else:
            count_idx = 0
            stop_idx = max_epochs
        population = new_population  
    return population, GA.X, GA.f_X, GA.max(population), idx
    # population = A.generation_change(mutation)


def run_GA_stop_condition_1_elitism(N, precision, ranges, f,max_epochs=1000, p_c=0.8, p_m=0.1):
    GA = GeneticAlgorithm(N=N, f=f, ranges=ranges, precision=precision)
    # A = GeneticAlgorithm(N=40, filename="123.txt")
    # GeneticAlgorithm
    maxes = []
    means = []

    rate = 0
    idx = 0
    population = GA.create_population()
#     GA.plot_data()
    # print(population)
    stop_idx = max_epochs
    count_idx = 0
    while idx < max_epochs:
        idx += 1
        maxes.append(GA.max(population))
        means.append(GA.mean(population))
        f_population = GA.evaluate_population(population)
        ranges = GA.create_ranges(f_population)
        binary_population = GA.int_to_binary(population)
        parents = GA.choose_parents(ranges, binary_population)

        children = GA.crossover(parents, 0.9)
        mutation = GA.mutation(children, 0.1)

        new_population = GA.binary_to_int(mutation)
        new_population = GA.elitism(population, new_population)
        rate = GA.child_parent_changes(population, new_population)
        if GA.max(population) == GA.max(new_population):
            if stop_idx == max_epochs:
                stop_idx = int((idx+20) * 0.9)
            count_idx += 1
            if count_idx == stop_idx:
                break
        else:
            count_idx = 0
            stop_idx = max_epochs
        population = new_population  
    return population, GA.X, GA.f_X, GA.max(population), idx  
    
def run_GA_stop_condition_2(N, precision, ranges, f,max_epochs=1000, p_c=0.8, p_m=0.1):
    GA = GeneticAlgorithm(N=N, f=f, ranges=ranges, precision=precision)
    # A = GeneticAlgorithm(N=40, filename="123.txt")
    # GeneticAlgorithm
    maxes = []
    means = []

    rate = 0
    idx = 0
    population = GA.create_population()
#     GA.plot_data()
    # print(population)
    stop_idx = max_epochs
    count_idx = 0
    while idx < max_epochs:
        idx += 1
        maxes.append(GA.max(population))
        means.append(GA.mean(population))
        f_population = GA.evaluate_population(population)
        ranges = GA.create_ranges(f_population)
        binary_population = GA.int_to_binary(population)
        parents = GA.choose_parents(ranges, binary_population)

        children = GA.crossover(parents, 0.5)
        mutation = GA.mutation(children, 0.1)

        new_population = GA.binary_to_int(mutation)
        new_population = GA.elitism(population, new_population)
        rate = GA.child_parent_changes(population, new_population)
        if rate > 0.3:
            if stop_idx == max_epochs:
                stop_idx = int((idx+5) * 0.4 )
            count_idx += 1
            if count_idx == stop_idx:
                break
        else:
            count_idx = 0
            stop_idx = max_epochs
        population = new_population  
    return population, GA.X, GA.f_X, GA.max(population), idx


def run_GA_stop_condition_2_elitism(N, precision, ranges, f,max_epochs=1000, p_c=0.8, p_m=0.1):
    GA = GeneticAlgorithm(N=N, f=f, ranges=ranges, precision=precision)
    # A = GeneticAlgorithm(N=40, filename="123.txt")
    # GeneticAlgorithm
    maxes = []
    means = []

    rate = 0
    idx = 0
    population = GA.create_population()
#     GA.plot_data()
    # print(population)
    stop_idx = max_epochs
    count_idx = 0
    while idx < max_epochs:
        idx += 1
        maxes.append(GA.max(population))
        means.append(GA.mean(population))
        f_population = GA.evaluate_population(population)
        ranges = GA.create_ranges(f_population)
        binary_population = GA.int_to_binary(population)
        parents = GA.choose_parents(ranges, binary_population)

        children = GA.crossover(parents, 0.5)
        mutation = GA.mutation(children, 0.1)

        new_population = GA.binary_to_int(mutation)
        # new_population = GA.elitism(population, new_population)
        rate = GA.child_parent_changes(population, new_population)
        if rate > 0.3:
            if stop_idx == max_epochs:
                stop_idx = int((idx+5) * 0.4 )
            count_idx += 1
            if count_idx == stop_idx:
                break
        else:
            count_idx = 0
            stop_idx = max_epochs
        population = new_population  
    return population, GA.X, GA.f_X, GA.max(population), idx


def plot_all(P, X, f_X, M, ax=plt):
    ax.plot(X, f_X)
    ax.plot(X[P], f_X[P], 'y*')
    ax.plot(X[M], f_X[M], 'ro')
