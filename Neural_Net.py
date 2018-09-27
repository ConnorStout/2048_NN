import numpy as np
import Game as G


class Neural_Net:
    test = []
    def __init__(self):
        print('')
        self.test = []

    def initialize_parameters(size):
        parameters = {}
        for l in range(1, len(size)):
            parameters['W' + str(l)] = np.random.randn(size[l], size[l - 1]) * 0.01
            parameters['b' + str(l)] = np.zeros((size[l], 1))

        return parameters

    def sigmoid(self,Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self,Z):
        return np.maximum(0, Z)

    def linear_forward(self,A, W, b):
        return np.dot(W, A) + b

    def linear_activation_forward(self,A_prev, W, b, activation):
        if activation == "sigmoid":

            Z = self.linear_forward(A_prev, W, b)
            A = self.sigmoid(Z)

        elif activation == "relu":

            Z = self.linear_forward(A_prev, W, b)
            A = self.relu(Z)
        return A

    def L_model_forward(self,X, parameters):
        A = X
        L = len(parameters) // 2

        for l in range(1, L):
            A_prev = A
            A = self.linear_activation_forward(A_prev,parameters['W' + str(l)],parameters['b' + str(l)],activation='relu')

        return self.linear_activation_forward(A,parameters['W' + str(L)],parameters['b' + str(L)],activation='sigmoid')



    def getFlattenedArr(self,array):
        a = np.array(array).flatten('F').reshape(16, 1)
        for x in range(0,16):
            if(a[x]!=0):
                a[x] = np.log2(a[x])
        return np.divide(a,max(a))

    def run_net(self,size,parameters):
        q = G.Game()
        q.restart()
        # print('here is start ')
        # q.print_board()
        #q.print_board()
        prev_board = self.getFlattenedArr(q.board)

        while(q.finished == False):
            AL = self.L_model_forward(prev_board, parameters)
            one_d = [i[0] for i in AL]
            #print(one_d)
            maxpos = one_d.index(max(one_d))

            check = q.take_turn(maxpos)
            if(check == False):
                #print('turn failed')
                    q.endStep()
                    break

            #print('new_board = ' + str(list(np.array(q.board).ravel())))
            new_board = self.getFlattenedArr(q.board)
            prev_board = new_board
        prev_board = []

        return q.score

