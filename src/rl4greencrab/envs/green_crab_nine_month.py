class greenCrabNineMonth(greenCrabEnv):
    def __init__(self, config={}):
        super().__init__(config=config)
        self.observation_space = spaces.Box(
            np.float32(9 * [-1]),
            np.float32(9 * [1]),
        )
        self.action_space = spaces.Box(
            np.float32(9 * [-1]),
            np.float32(9 * [1]),
        )

    def reset(self):
        unnorm_obs, info = super().reset(self)
        norm_obs = 2 * unnorm_obs / self.max_observation - 1
        return norm_obs, info

    def step(self,action):
        #size selective harvest rate, given action
        harvest_rate = np.array(
            [
                1 - np.exp( -(
                    self.size_sel_log(
                        self.traps_pmax, 
                        self.traps_midpoint, 
                        self.traps_k
                    ) * (1+action[i]) / 2
                ))
                for i in range(self.ntime)
            ]
        ) # n_size columns, ntime rows

        #add pop at t=1
        size_freq = np.zeros(shape=(self.nsize,self.ntime),dtype='object')
        size_freq[:,0] = self.state
        
        #create array to store # removed
        removed = np.zeros(shape=(self.nsize,self.ntime),dtype='object')
        
        #calculate removed and record observation at t=0
        #apply monthly harvest rate
        removed[:,0] = [np.random.binomial(size_freq[k,0], harvest_rate[k, 0]) for k in range(self.nsize)]
            
        #record the catch in the observation space
        self.observations[0] = np.sum(removed[:,0])

        
        #loop through intra-annual change (9 total months), t=2+        
        for j in range(self.ntime-1):
            #project to next month
            n_j = self.gm_ker@(size_freq[:,j] - removed[:,j])

            size_freq[:,j+1] = [np.random.binomial(n=n_j[k], p=self.pmort) for k in range(self.nsize)]
            removed[:,j+1] = [np.random.binomial(size_freq[k,j+1], harvest_rate[k,j+1]) for k in range(self.nsize)]
            
        self.observations = np.array([np.sum(removed[:,j]) for j in range(self.ntime)], dtype = np.float32)


        #calculate new adult population after overwinter mortality
        new_adults = [ np.random.binomial(size_freq[k,8],self.w_mort_exp[k]) for k in range(self.nsize) ]

        #simulate new recruits
        local_recruits = np.random.normal(self.dd_growth(size_freq[:,self.ntime-1]),self.env_stoch)
        nonlocal_recruits = np.random.poisson(self.imm)*(1-np.sum(size_freq[:,self.ntime-1])/self.K)
        recruit_total = local_recruits + nonlocal_recruits

        #get sizes of recruits
        recruit_sizes = (norm.cdf(self.bndry[1:(self.nsize+1)],self.init_mean_recruit,self.init_sd_recruit)-\
         norm.cdf(self.bndry[0:self.nsize],self.init_mean_recruit,self.init_sd_recruit))*recruit_total

        #store new population size (and cap off at zero pop)
        self.state = np.maximum(recruit_sizes + new_adults, 0)

        #calculate reward
        self.reward = self.reward_func(action)
        self.years_passed += 1

        done = bool(self.years_passed > self.Tmax)

        # if np.sum(self.state) <= 0.001:
        #     done = True

        return self.observations, self.reward, done, done, {}
        
        