import numpy as np 
class HMM:

    def __init__(self, N=15, T_prob=None, B_list=None):
        '''
        Args:
            N: The grid side length
            T_prob: list of [p_right, p_left, p_up, p_down, p_same]
            B_list: contains 4 Bs correponding to 4 sensors
                each B is 1d np array of shape: B: [N^2]
                the sensor matrices are in order of S1, S2, S3, S4
        '''

        
        #side length
        self.N = N 

        #number of states
        self.n = self.get_num_hidden_states()

        #the hidden states 
        self.states = self.get_all_hidden_states()
        
        self.hidden_state_idx = self.get_hidden_state_index_map()
        
        #[n x n]
        # self.T_prob = T_prob
        self.T_prob = self.set_T_prob()
        self.T = self.create_T(self.T_prob) 
        
        #the sensor index -> sensor symbol
        self._sensor_symbol = self.create_sensor_symbol()
        
        #the sensor grid
        # self.sensor_grid = [[1,9,1,9], [7,15,1,9], [7,15,7,15], [1,9,7,15]]
        self.sensor_grid = [[1,9,1,9], [1,9,7,15], [7,15,7,15], [7,15,1,9]]

        #[n x 16]
        # self.B_list = B_list
        self.B_list = self.set_B_list()
        self.B = self.create_B(self.B_list)

        #[n]
        self.rho = np.zeros(self.n)
        self.rho[0] = 1 
        
        self.hidden_state_idx = self.create_hidden_state_index_map()
        
        assert N > 0
        assert len(self.T_prob) == 5
        assert len(self.B_list) == 4
        for B in self.B_list:
            assert B.shape[0] == self.N*self.N 
    
    def get_num_hidden_states(self):
        return (self.N**2)
    
    def get_all_hidden_states(self):
        states = []
        for i in range(self.n):
            
            # ith state
            x = (i % self.N) + 1 
            y = (i // self.N) + 1
                        
            states.append((x,y))
                
        return states
    
    def create_sensor_symbol(self):
        '''
        This function returns back the sensor symbols in the form of [S1, S2, S3, S4]
        where S1, S2, S3, S4 are the sensor outputs. There are a total of 2**4 = 16 observation 
        states. Every observation state might look like [0, 0, 1, 1].
        '''
        sensor_symbol = []
        for s in range(16):
            symbol = []
            for i in range(4):
                symbol.append(s // int(2**(4-i-1)))
                s = s % (int(2**(4-i-1)))
            sensor_symbol.append(symbol)
        return np.array(sensor_symbol)
    
    def set_B_list(self, b_list=None):
        if b_list==None:
            b_list = []
            for i in range(4):
                b = []
                for j in range(self.n):
                    x,y = self.states[j]
                    if i == 0:
                        if x in range(1, 10) and y in range(1, 10):
                            b_ij = (18-(x-1)-(y-1))/18
                        else:
                            b_ij = 0
                    elif i == 3:
                        if x in range(1, 10) and y in range(7, 16):
                            b_ij = (18-(x-1)+(y-15))/18
                        else:
                            b_ij = 0
                    elif i == 2:
                        if x in range(7, 16) and y in range(7, 16):
                            b_ij = (18+(x-15)+(y-15))/18
                        else:
                            b_ij = 0
                    else:
                        if x in range(7, 16) and y in range(1, 10):
                            b_ij = (18+(x-15)-(y-1))/18
                        else:
                            b_ij = 0
                    b.append(b_ij)
                b_list.append(b)
            self.B_list = np.array(b_list)
        return self.B_list

    def get_B_list(self):
        return self.B_list
    
    #create the matrix
    def create_B(self, B_list):
    # def create_B(self):
        '''
        Args:
            B_list: list of 4 sensor Bs of shape [self.n]
        Return:
            B: [n,16]
        '''
        if B_list is None:
            B_list = self.set_B_list()
            
        B = np.ones((self.n, 16))
        for j in range(16):
            s = self._sensor_symbol[j] # the binary representation for the observation state j
            for i, k in enumerate(s):
                B[:,j] *= (k*B_list[i] + (1-k)*(1-B_list[i])) # this part is not yet clear, need to check it out
        return B
    
    def set_T_prob(self, pr=0.4, pl=0.1, pu=0.3, pd=0.1, ps=0.1) :
        T_prob = [pr, pl, pu, pd, ps]
        return T_prob
        
    def get_T_prob(self):
        return self.T_prob
    
    def create_T(self, T_prob):
    # def create_T(self):
        if T_prob is None:
            T_prob = self.set_T_prob(0.4, 0.1, 0.3, 0.1, 0.1)
            
        #get the probabilities
        pr, pl, pu, pd, ps = T_prob[0], T_prob[1], T_prob[2], T_prob[3], T_prob[4]
        T = np.zeros((self.n, self.n))
        for i in range(self.n):
            
            x = (i % self.N) + 1 
            y = (i // self.N) + 1
            
            #self transition
            T[i,i] = ps

            #move to right
            if(x + 1 <= self.N):
                T[i,i+1] = pr
            else:
                T[i,i] += pr
            
            #move left
            if(x - 1 > 0):
                T[i,i-1] = pl
            else:
                T[i,i] += pl
            
            #move up
            if(y + 1 <= self.N):
                T[i,i+self.N] = pu
            else:
                T[i,i] += pu

            #move down
            if(y - 1 > 0):
                T[i,i-self.N] = pd
            else:
                T[i,i] += pd
        return T

    def create_hidden_state_index_map(self):
        # idx = np.arange(self.n)
        hidden_state_idx = {}
        for i, k in enumerate(range(self.n)):
            x,y = self.states[k]
            hidden_state_idx[(x,y)] = i
            self.hidden_state_idx = hidden_state_idx
        return self.hidden_state_idx
        
    def get_hidden_state_index_map(self):
        self.hidden_state_idx = self.create_hidden_state_index_map()
        return self.hidden_state_idx
        
    def sensor_mask(self, hidden_state, sensor_no):
        hidden_state_index = self.hidden_state_idx[hidden_state]
        if self.B_list[sensor_no][hidden_state_index]:
            return 1
        else:
            return 0
    
    def sample_observation_state_sequence(self, hidden_sate_sequence):
        obs_seq = []
        for hidden_state in hidden_sate_sequence:
            obs_si = []
            for i in range(4):
                obs_si.append(self.sensor_mask(hidden_state, i)) 
            obs_seq.append(obs_si)
        # print(obs_seq)
        return obs_seq

    
    def sample_hidden_state_sequence(self, t, seed=42):
        '''
        Args:
            t: integer, trajectory length
        '''
        np.random.seed(seed)
        
        def neighbourhood(s_i):
            x, y = s_i

            neigh = [(x,y)]
            
            #move to right
            if(x + 1 <= self.N):
                neigh.append((x+1, y))
            
            #move left
            if(x - 1 > 0):
                neigh.append((x-1, y))

            #move up
            if(y + 1 <= self.N):
                neigh.append((x, y+1))

            #move down
            if(y - 1 > 0):
                neigh.append((x, y-1))

            return neigh

        s_seq = []
        
        # at Time=0
        s_1 = np.random.choice(a = list(self.hidden_state_idx.values()), size=1, p=self.rho).item()
        s_seq.append(self.states[s_1])
  
        # Time=1 to Time=t-1
        for i in range(1, t):
            prev_hidden_state_idx = self.hidden_state_idx[s_seq[i-1]]
            neighs_si = neighbourhood(s_seq[i-1])
            neighs_si_idx = [self.hidden_state_idx[k] for k in neighs_si]
            p = [self.T[prev_hidden_state_idx, j] for j in neighs_si_idx]
            s_i = np.random.choice(a = neighs_si_idx, size=1, p=p).item()
            s_seq.append(self.states[s_i])
            
            
        return s_seq
        
    def obs_to_idx(self, obs_i):
        binary_string = ''.join(str(bit) for bit in obs_i)
        integer_value = int(binary_string, 2)
        return integer_value
    
    def forward_inference(self, obs):
        '''
        Args:
            obs: np.array of shape [T]
        '''
        T = len(obs)
        obs_0_idx = self.obs_to_idx(obs[0])
        
        forward = np.zeros((self.n, T))

        # initializing all the states at time t=0
        forward[:, 0] = (forward[:, 0] + self.rho).dot(np.diag(self.B[:, obs_0_idx]))
        # from time t=1 to t=T-1
        for t in range(1, T):
            obs_t_idx = self.obs_to_idx(obs[t])
            forward[:, t] = (forward[:, t-1].dot(self.T)).dot(np.diag(self.B[:, obs_t_idx]))
        forward_prob = np.sum(forward[:, T-1])
        return forward_prob
    
    
    def forward(self, obs):
        '''
        Args:
            obs: np.array of shape [T]
        Return:
            alpha: np.array of shape [T, self.n]
        '''
        T = len(obs)
        obs_0_idx = self.obs_to_idx(obs[0])
        
        alpha = np.zeros((self.n, T))

        # Initialize alpha at time t=0
        alpha[:, 0] = self.rho * self.B_hat[:, obs_0_idx]

        # Compute alpha for t=1 to t=T-1
        for t in range(1, T):
            obs_t_idx = self.obs_to_idx(obs[t])
            alpha[:, t] = (alpha[:, t-1].dot(self.T_hat)) * self.B_hat[:, obs_t_idx]

        return alpha.T  

    
    def backward(self, obs):
        '''
        Args:
            obs: np.array of shape [T]
        Return:
            beta: np.array of shape [T, self.n]
        '''
        T = len(obs)  
        n = self.n  

        beta = np.zeros((n, T))  

        # Initialize the last column of beta (at time t = T-1)
        beta[:, T-1] = 1 

        # Backward pass: from time t = T-2 to t = 0
        for t in range(T-2, -1, -1):
            obs_t1_idx = self.obs_to_idx(obs[t+1])  

            beta[:, t] = self.T_hat.dot(self.B_hat[:, obs_t1_idx] * beta[:, t+1])

        return beta.T

        

    def e_step(self, alpha, beta, obs):
        '''
        Args:
            alpha: np.array of shape [R, T, n] - forward probabilities
            beta: np.array of shape [R, T, n] - backward probabilities
        Return:
            gamma: np.array of shape [R, T, n] - expected state probabilities
            ksi: np.array of shape [R, T-1, n, n] - expected state transition probabilities
        '''
        R, T, _ = alpha.shape
        
        gamma = np.zeros((R, T, self.n))
        for r in range(0, R):
            gamma_denom = np.sum(alpha[r, T-1, :])
            for t in range(0, T):
                for i in range(0, self.n):
                    gamma[r, t, i] += (alpha[r, t, i]*beta[r, t, i])/gamma_denom
        
        ksi = np.zeros((R, T, self.n, self.n))    
        for r in range(0, R):
            ksi_denom = np.sum(alpha[r, T-1, :])
            for t in range(0, T-1):
                obs_t1_idx = self.obs_to_idx(obs[r][t+1])
                for i in range(0, self.n):
                    x = (i % self.N) + 1 
                    y = (i // self.N) + 1
                    
                    #self transition
                    ksi[r, t, i, i] += (alpha[r, t, i]*self.T_hat[i, i]*self.B_hat[i, obs_t1_idx]*beta[r, t+1, i])/ksi_denom
                    
                    #move to right
                    if(x + 1 <= self.N):
                        ksi[r, t, i, i+1] += (alpha[r, t, i]*self.T_hat[i, i+1]*self.B_hat[i+1, obs_t1_idx]*beta[r, t+1, i+1])/ksi_denom

                    #move left
                    if(x - 1 > 0):
                        ksi[r, t, i, i-1] += (alpha[r, t, i]*self.T_hat[i, i-1]*self.B_hat[i-1, obs_t1_idx]*beta[r, t+1, i-1])/ksi_denom
                    
                    #move up
                    if(y + 1 <= self.N):
                        ksi[r, t, i, i+self.N] += (alpha[r, t, i]*self.T_hat[i, i+self.N]*self.B_hat[i+self.N, obs_t1_idx]*beta[r, t+1, i+self.N])/ksi_denom

                    #move down
                    if(y - 1 > 0):
                        ksi[r, t, i, i-self.N] += (alpha[r, t, i]*self.T_hat[i, i-self.N]*self.B_hat[i-self.N, obs_t1_idx]*beta[r, t+1, i-self.N])/ksi_denom
                    
        return gamma, ksi
    
    def m_step(self, ksi, gamma, obs, B_trainable):
        R, T, _ = gamma.shape
        
        # UPDATE T_hat
        ps = np.zeros((self.n,))  # for stationary transitions
        pr = np.zeros((self.n,))  # for right transitions
        pl = np.zeros((self.n,))  # for left transitions
        pu = np.zeros((self.n,))  # for up transitions
        pd = np.zeros((self.n,))  # for down transitions
        denom = np.zeros((self.n,))  # for normalization
        
        for i in range(self.n):
            x = (i % self.N) + 1 
            y = (i // self.N) + 1       
                 
            denom[i] += np.sum(ksi[:, :, i, i])
            
            ps[i] = np.sum(ksi[:, :, i, i])
            
            # Handle right transition
            if x + 1 <= self.N:
                pr[i] = np.sum(ksi[:, :, i, i+1])
                denom[i] += np.sum(ksi[:, :, i, i+1])

            # Handle left transition
            if x - 1 > 0:
                pl[i] = np.sum(ksi[:, :, i, i-1])
                denom[i] += np.sum(ksi[:, :, i, i-1])

            # Handle up transition
            if y + 1 <= self.N:
                pu[i] = np.sum(ksi[:, :, i, i+self.N])
                denom[i] += np.sum(ksi[:, :, i, i+self.N])

            # Handle down transition
            if y - 1 > 0:
                pd[i] = np.sum(ksi[:, :, i, i-self.N])
                denom[i] += np.sum(ksi[:, :, i, i-self.N])
        
        pr = np.sum(pr)/np.sum(denom)
        pl = np.sum(pl)/np.sum(denom)
        pu = np.sum(pu)/np.sum(denom)
        pd = np.sum(pd)/np.sum(denom)
        ps = np.sum(ps)/np.sum(denom)

        T_prob = [pr, pl, pu, pd, ps]
        
        self.T_hat = self.create_T(T_prob)
        
        # UPDATE self.B_hat        
        if B_trainable:
            obs = np.asarray(obs)
            for i in range(self.n):
                denominator = np.sum(gamma[:, :, i])
                numerator = np.zeros(16)
                for t in range(T):
                    for k in range(16): 
                        indicator_mask = (np.array(list(map(self.obs_to_idx, obs[:, t]))) == k).astype(float)
                        numerator[k] += np.sum(gamma[:, t, i] * indicator_mask)

            # Convert to log-space for numerical stability
            with np.errstate(divide='ignore', invalid='ignore'):
                log_numerator = np.log(numerator + 1e-26)  # Add a small constant to avoid log(0)
                log_denominator = np.log(denominator + 1e-26)  # Add a small constant to avoid log(0)
                
                # Compute log-space probabilities
                log_B_hat = log_numerator - log_denominator
                self.B_hat[i, :] = np.exp(log_B_hat)
        
        
        return self.T_hat, self.B_hat


    def baum_welch(self, obs, n_iter=20, B_trainable=False):
        '''
        Args:
            obs: np.array of shape [R, T] - observation sequences
            n_iter: number of iterations for the Baum-Welch algorithm
        Return:
            T: final transition matrix of shape [n, n]
            B: final emission matrix of shape [n, 16]
            rho: final initial state distribution of shape [n]
        '''
        R, _, _ = obs.shape
        
        T_prob = [1/5, 1/5, 1/5, 1/5, 1/5]
        self.T_hat = self.create_T(T_prob) 
        print(self.T_hat.shape)
        
        self.B_hat = np.ones((self.n, 16)) * (1/16)
        
        avgKL_T_list = []
        avgKL__B_list = []
        for itr in range(n_iter):
            alpha_list = []
            beta_list = []
            
            for r in range(0, R):
                alpha_r = self.forward(obs[r])
                beta_r = self.backward(obs[r])
                alpha_list.append(alpha_r)
                beta_list.append(beta_r)
            
            alpha = np.array(alpha_list)
            beta = np.array(beta_list)
            print(f"Iter: {itr} ; alpha and beta done")
            # E-step: compute gamma and ksi
            gamma, ksi = self.e_step(alpha, beta, obs)
            print(f"Iter: {itr} ; E step done")
            # M-step: update T, B
            self.T_hat, self.B_hat = self.m_step(ksi, gamma, obs, B_trainable)
            print(f"Iter: {itr} ; M step done")
   
            avgKL_T = self.kl_divergence(self.T, self.T_hat)
            avgKL_B = self.kl_divergence(self.B, self.B_hat)
            
            print(f"Iter: {itr} ; avgKL_T: {avgKL_T} ; avgKL_B: {avgKL_B}")
            
            avgKL_T_list.append(avgKL_T)
            avgKL__B_list.append(avgKL_B)

        
        return self.T_hat, self.B_hat, self.rho, avgKL_T_list, avgKL__B_list
    
    
    def viterbi(self, obs):
        '''
        Args:
            obs: [R, T]
        '''
        T = len(obs)
        obs_0_idx = self.obs_to_idx(obs[0])
        
        viterbi = np.zeros((self.n, T))
        backpointer = np.zeros((self.n, T))
        
        # initializing all the states at time t=0
        viterbi[:, 0] = (viterbi[:, 0] + self.rho).dot(np.diag(self.B[:, obs_0_idx]))
        backpointer[:, 0] = backpointer[:, 0] #- np.ones(backpointer[:, 0].shape)
        
        # from time t=1 to t=T-1
        for t in range(1, T):
            obs_t_idx = self.obs_to_idx(obs[t])
            viterbi[:, t] = np.max((np.diag(viterbi[:, t-1]).dot(self.T)) * (np.tile(self.B[:, obs_t_idx], (self.n, 1)).T), axis=0)
            backpointer[:, t] = np.argmax((np.diag(viterbi[:, t-1]).dot(self.T)) * (np.tile(self.B[:, obs_t_idx], (self.n, 1)).T), axis=0)
        
        best_path_prob = np.max(viterbi[:, T-1])
        best_path_pointer = np.argmax(viterbi[:, T-1])
        
        best_path = [best_path_pointer]
        next_pointer = best_path_pointer
        for t in range(T-2, -1, -1):
            next_pointer = backpointer[int(next_pointer), t]
            best_path.append(int(next_pointer))

        best_path = best_path[::-1]
        best_path = list(map(lambda x : self.states[x], best_path))
        return best_path

    def kl_divergence(self, true_matrix, estimated_matrix):
        '''
        Args:
            true_matrix: np.array of shape [n, n] or [n, k] - true transition or emission matrix
            estimated_matrix: np.array of shape [n, n] or [n, k] - estimated transition or emission matrix
        Returns:
            avg_kl: float - the average KL divergence between the true and estimated matrices
        '''

        kl_div = 0
        for i in range(true_matrix.shape[0]):
            for j in range(true_matrix.shape[1]):
                if estimated_matrix[i,j] and true_matrix[i,j]:
                    kl_div += true_matrix[i, j]*(np.log(true_matrix[i,j]) - np.log(estimated_matrix[i,j]))
        avg_kl_div = kl_div/true_matrix.shape[0]        
                 
        return avg_kl_div
