from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(X_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    # TODO
    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array delta[i, t] = P(X_t = s_i, Z_1:Z_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])

        #print("S: " + str(S)) # S = num_state
        #print("L: " + str(L))
        ###################################################
        # Edit here

        print(self.state_dict)
        print(self.obs_dict)

        for i in self.state_dict:
            s = self.state_dict[i]
            alpha[s][0] = self.pi[s] * self.B[s][self.obs_dict[Osequence[0]]]

        
        for t in range(1, L):
            for i in self.state_dict:
                s = self.state_dict[i]
                zt = self.obs_dict[Osequence[t]]

                alpha[s][t] = self.B[s][zt] * np.sum([self.A[self.state_dict[s_p]][s] * alpha[self.state_dict[s_p]][t-1] for s_p in self.state_dict])
        



        ###################################################
        return alpha

    # TODO:
    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array gamma[i, t] = P(Z_t+1:Z_T | X_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################

        T = L - 1

        for i in self.state_dict:
            s = self.state_dict[i]
            beta[s][T] = 1

        timeSteps = range(T)
        for t in timeSteps[::-1]:
            for i in self.state_dict:
                s = self.state_dict[i]
                ztp1 = self.obs_dict[Osequence[t + 1]]
                beta[s][t] = np.sum([self.A[s][self.state_dict[s_p]] * self.B[self.state_dict[s_p]][ztp1] * beta[self.state_dict[s_p]][t + 1] for s_p in self.state_dict])


        ###################################################
        return beta

    # TODO:
    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(Z_1:Z_T | λ)
        """
        prob = 0
        ###################################################
        L = Osequence.shape[0]
        T = L - 1

        alpha = self.forward(Osequence)

        for i in self.state_dict:
            s = self.state_dict[i]
            prob += alpha[s][T]


        ###################################################
        return prob

    # TODO:
    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(X_t = i | O, λ)
        """
        prob = 0
        ###################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq = self.sequence_prob(Osequence)

        """
        N = alpha.shape[0]
        L = alpha.shape[1]
        prob = np.zeros((N, L))        
        for t in range(L):
            for s in self.state_dict:
                i = self.state_dict[s]
                num = alpha[i][t] * beta[i][t]
                prob[i][t] = num / seq
        """

        prob = np.multiply(alpha, beta) / seq

        ###################################################
        return prob

    # TODO:
    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        N = len(self.pi)
        L = len(Osequence)
        T = L

        path_idx = np.zeros(T)

        sigma = np.zeros((N, L))
        delta = np.zeros((N, L))

        """
        for i in self.state_dict:
            s = self.state_dict[i]
            sigma[s][0] = self.pi[s] * self.B[s][self.obs_dict[Osequence[0]]]
        """
        
        for s in range(N):
            sigma[s][0] = self.pi[s] * self.B[s][self.obs_dict[Osequence[0]]]

        """
        for t in range(1, T):
            for i in self.state_dict:
                s = self.state_dict[i]
                zt = self.obs_dict[Osequence[t]]

                sigma[s][t] = self.B[s][zt] * np.max([self.A[self.state_dict[s_p]][s] * sigma[self.state_dict[s_p]][t-1] for s_p in self.state_dict])
                delta[s][t] = np.argmax([self.A[self.state_dict[s_p]][s] * sigma[self.state_dict[s_p]][t-1] for s_p in self.state_dict])
        """

        for t in range(1, T):
            for s in range(N):
                zt = self.obs_dict[Osequence[t]]

                sigma[s][t] = self.B[s][zt] * np.max([self.A[s_p][s] * sigma[s_p][t-1] for s_p in range(N)])
                delta[s][t] = np.argmax([self.A[s_p][s] * sigma[s_p][t-1] for s_p in range(N)])
        

        
        path_idx[T-1] = np.argmax(sigma, axis=0)[T-1]

        timeSteps = range(0, T - 1)
        for t in timeSteps[::-1]:
            path_idx[t] = delta[int(path_idx[t+1])][t+1]

        path = []

        keys=list(self.state_dict.keys())
        values=list(self.state_dict.values())

        for i, p in enumerate(path_idx):
            index = values.index(p)
            path.append(keys[index])

        


        ###################################################
        return path
