import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import gym

class Ising():
    ''' Simulating the Ising model
        Taken from https://rajeshrinet.github.io/blog/2014/ising-model/ 

       Critical temperature is T_c=(2J)/(k*ln(1+sqrt(2))) for the 2d case
    '''

        
    def __init__(self,beta=1/4,N=64):
        self.beta = beta
        self.N = N

    def reset(self):
        config = 2*np.random.randint(2, size=(self.N,self.N))-1
        self.config = config
        return config
        
    ## monte carlo moves
    def mcmove(self, config, N, beta):
        ''' This is to execute the MC moves using 
        Metropolis algorithm such that detailed
        balance condition is satisified'''
        for i in range(N):
            for j in range(N):            
                    a = np.random.randint(0, N)
                    b = np.random.randint(0, N)
                    s =  config[a, b]
                    nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
                    cost = 2*s*nb
                    if cost < 0:	
                        s *= -1
                    elif rand() < np.exp(-cost*beta):
                        s *= -1
                    config[a, b] = s
        return config
    
    def simulate_n(self,n_steps):   
        ''' This module simulates the Ising model'''
        for i in range(n_steps):
            self.mcmove(self.config, self.config.shape[0], self.beta)
        return self.config
                 
                    
    def configPlot(self, f, config, i, N, n_):
        ''' This modules plts the configuration once passed to it along with time etc '''
        X, Y = np.meshgrid(range(N), range(N))
        sp =  f.add_subplot(3, 3, n_ )  
        plt.setp(sp.get_yticklabels(), visible=False)
        plt.setp(sp.get_xticklabels(), visible=False)      
        plt.pcolormesh(X, Y, config, cmap=plt.cm.RdBu)
        plt.title('Time=%d'%i); plt.axis('tight')    
        plt.show()

class IsingEnv(gym.Env):

    def __init__(self,T=32,k=5):
        self.BETA_MIN = 0.01
        self.BETA_MAX = 0.3
        self.N = 64
        self.MAX_T = T

        self.k = k
        self.T = T

        self.betas = np.linspace(self.BETA_MIN,self.BETA_MAX,k)
        self.models = [Ising(beta=beta,N=self.N) for beta in self.betas]

        self.action_space = gym.spaces.Discrete(k)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(1, self.N, self.N),
            dtype=np.uint8,
        )

        self.reset()

    def _rescale_state(self,state):
        state = (state+1)/2 # between 0 and 1
        noise = np.random.uniform(0,0.1,size=state.shape)
        state = state + noise
        state = ( state / (state.max()) * 255. ).astype(np.uint8)
        return state

    def _generate_goal(self):
        goal_beta = np.random.uniform(self.BETA_MIN,self.BETA_MAX,1)

        self.goal_beta = goal_beta

        self.goal_model = Ising(beta=goal_beta,N=self.N)
        self.goal_model.reset()

        T = np.random.randint(0,self.MAX_T)

        self.goal = self.goal_model.simulate_n(T).copy()

    def reset(self):
        self._generate_goal()
        self.current_path = np.random.randint(0,self.k,1)
        state = None
        for i in range(len(self.models)):
            if i == self.current_path:
                state = self.models[i].reset()
            else:
                self.models[i].reset()
        self.t = 0
        return self._rescale_state(state)

    def step(self,action):
        self.current_path = action
        state = None
        for i in range(len(self.models)):
            if i == self.current_path:
                state = self.models[i].simulate_n(1).copy()
            else:
                self.models[i].simulate_n(1)
           
        reward = -np.log(np.linalg.norm(state-self.goal,ord=2)) # when ||s-goal||=0, reward=infty
        self.t += 1
        done = bool( self.t < self.T )
        return self._rescale_state(state), reward, done, {}

if __name__ == '__main__':
    env = IsingEnv(T=32,k=5)
    state = env.reset()
    a = env.action_space.sample()
    s,r,done,_ = env.step(a)
    print(s)