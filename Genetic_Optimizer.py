import Neural_Net as net
import numpy as np
import random as rand
import copy
import time
import Input as inp
class GeneticOptimizer:
    mutation_rate = 0.1
    cross_over_rate = 0.10
    generation_size = 100
    nn = net.Neural_Net()
    generation = 0
    score_progression = []
    generation_score_list = []
    generation_list = []
    average_list = []
    size = []
    def __init__(self,size,generation_size):
        prevs = inp.Input()
        self.size = size
        self.generation_size = generation_size
        # for x in range(0,generation_size):
        #     self.generation_list.append(self.initialize_parameters(size))
        self.generation_list = prevs.prev

    def run_generation(self):
        begin = time.time()
        for x in self.generation_list:
            self.generation_score_list.append(self.nn.run_net(self.size,x))
            #self.nn.
            #print('next')
        #print(self.generation_score_list)
        print('time to run gen = ' + str(time.time()-begin))
        print(max(self.generation_score_list))
    def initialize_parameters(self,layer_dims):
        np.random.seed()
        parameters = {}
        L = len(layer_dims)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1])
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

            assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters
    def produce_next_generation(self):
        begin = time.time()
        next_gen_list =[]
        sorted_list = list(reversed(sorted(self.generation_score_list)))
        #print(sorted_list)
        #print(self.generation_score_list)
        for x in sorted_list[:50]:
            index = self.generation_score_list.index(x)
            next_gen_list.append((copy.deepcopy(self.generation_list[index])))
        for x in range(0,50):
            random_A = rand.randint(0, 10)
            random_B = rand.randint(0, 10)
            #print(random_A)
            indexA = self.generation_score_list.index(sorted_list[random_A])
            indexB = self.generation_score_list.index(sorted_list[random_B])
            next_gen_list.append((self.perform_crossovers(copy.deepcopy(self.generation_list[indexA]),copy.deepcopy(self.generation_list[indexB]))))
        for x in sorted_list[:50]:
            for y in range(0, 3):
                index = self.generation_score_list.index(x)
                next_gen_list.append(self.perform_mutaions((copy.deepcopy(self.generation_list[index]))))
        #print('the important stuff' + str(self.generation_score_list))
        max_val = max(self.generation_score_list)
        self.score_progression.append(max_val)
        self.average_list.append(sum(self.generation_score_list)/self.generation_size)
        self.generation_score_list = []
        self.generation_list = next_gen_list
        print('time to produce next gen = ' + str(time.time() - begin))
    def perform_mutaions(self,input):
        new_random = rand.randint(0, 4)
        if (new_random == 2):
            for x in range(1,len(self.size)):

                w = input['W' + str(x)]
                for y in np.nditer(w, op_flags=['readwrite']):
                    if rand.randint(0, 20) == 2:
                        y[...] = y[...] + rand.uniform(-1, 1)
                b = input['b' + str(x)]
                rand_b = rand.uniform(-1, 1)
                if (rand.randint(0, 10) == 1):
                    for y in np.nditer(b, op_flags=['readwrite']):
                        y[...] = y[...] + rand_b
        return input

    def perform_crossovers(self, inputA, inputB):
        cross_type = rand.randint(1, 2)
        new_weights = self.initialize_parameters(self.size)
        #print(cross_type)
        if(cross_type ==2):
            for x in range(0,len(self.size)-1):

                randB = rand.randint(0, self.size[x+1])

                # print(inputA['b' + str(x + 1)][:randB])
                # print(inputB['b' + str(x + 1)][randB - 1:])
                for y in range(0,self.size[x+1]):
                    randW = rand.randint(0, self.size[x])
                    # print(inputA['W' + str(x + 1)][y][:randW])
                    # print(inputB['W' + str(x+1)][y][randW:])
                    # print(list(inputA['W' + str(x + 1)][y][:randW])+ list(inputB['W' + str(x+1)][y][randW:]))

                    new_weights['W' + str(x+1)][y] = list(inputA['W' + str(x + 1)][y][:randW])+ list(inputB['W' + str(x+1)][y][randW:])
                #new_weights['b' + str(x + 1)] = np.concatenate((inputA['b' + str(x + 1)][:randB],inputB['b' + str(x + 1)][randB:]),axis=0)
                new_weights['b' + str(x + 1)] = inputA['b' + str(x + 1)][:]
        else:
            #problem with both
            for x in range(0,len(self.size)-1):
                randW = rand.randint(0, self.size[x+1])
                randB = rand.randint(0, self.size[x+1])
                # print(inputA['W' + str(x + 1)][:randW])
                # print(inputB['W' + str(x + 1)][randW-1:])
                # print(inputA['b' + str(x + 1)][:randB])
                # print(inputB['b' + str(x + 1)][randB - 1:])

                new_weights['W' + str(x+1)] = np.concatenate((inputA['W' + str(x+1)][:randW], inputB['W' + str(x+1)][randW:]),axis=0)
                #new_weights['b' + str(x + 1)] = np.concatenate((inputA['b' + str(x + 1)][:randB],inputB['b' + str(x + 1)][randB:]),axis=0)
                new_weights['b' + str(x + 1)] = inputA['b' + str(x + 1)][:]
        return new_weights

    def print_weights(self,weighs):
        for x in range(0,2):
            print('W1: '+str(np.array(weighs[x]['W1']).flatten('F')))
            print('b1: '+ str(np.array(weighs[x]['b1']).flatten('F')))

gen = GeneticOptimizer([16,24,16,8,4],250)
begin = time.time()
#gen.print_weights(gen.generation_list[0])
#print(gen.print_weights(gen.generation_list))


for x in range(1,2000):
    print('generation ' + str(x)  + ' = ')
    gen.run_generation()
    #print(gen.generation_score_list)
    gen.produce_next_generation()

# print('generation ' + str(x)  + ' = ')
# gen.run_generation()

print(gen.generation_score_list)
print(gen.score_progression)
print(gen.average_list)
print('time for whole= ' + str(time.time()-begin))
f = open("dict.txt","w")
f.write(str(gen.generation_list))
f.close()
#print(gen.generation_list)


#time = 2181.880056142807


#[4, 0, 0, 0, 4, 0, 0, 16, 68, 4, 24, 72, 88, 0, 116, 0, 0, 0, 4, 0, 0, 16, 0, 0, 48, 8, 96, 8, 432, 212, 212, 0, 4, 96, 8, 0, 0, 0, 156, 204, 20, 0, 48, 0, 136, 80, 80, 12, 0, 0, 64, 20, 16, 0, 52, 20, 20, 0, 0, 212, 0, 0, 20, 4, 44, 8, 340, 8, 0, 112, 72, 8, 16, 88, 0, 0, 144, 0, 412, 44, 168, 36, 0, 44, 48, 72, 448, 16, 0, 4, 24, 120, 0, 24, 0, 4, 940, 24, 48, 4, 0, 80, 4, 96, 132, 0, 0, 28, 4, 128, 0, 16, 20, 132, 72, 304, 276, 4, 144, 0, 0, 0, 0, 4, 104, 220, 4, 300, 120, 28, 40, 0, 0, 0, 32, 16, 20, 52, 0, 216, 300, 0, 8, 0, 0, 24, 16, 200, 0, 32]
#time for whole= 25305.651768922806
