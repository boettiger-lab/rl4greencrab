from rl4greencrab import greenCrabSimplifiedEnv
import numpy as np


def test_action_units():
    env = greenCrabSimplifiedEnv()
    env.reset()
    action = np.array([-1,-1,-1]) # no traps
    natural_units = np.maximum( env.max_action * (1 + action)/2 , 0.)
    assert np.array_equal(natural_units, np.array([0,0,0])) # check if no crab exist and no change in population when do nothing

def test_no_harvest():
    env = greenCrabSimplifiedEnv()
    env.reset()
    
    steps = 10
    for i in range(steps):
        observation, rew, term, trunc, info  = env.step(np.array([-1,-1, -1]))
        assert info == {}
        assert trunc == False
        assert term == False
        assert rew < 0
        assert sum(env.state) > 0
        assert observation[0] == -1 and observation[1] == -1 #check if catch rate is zero in observation


def test_full_harvest():
    env = greenCrabSimplifiedEnv()
    env.reset()
    
    steps = env.Tmax
    prev_state = env.state # store the state before taking step
    for i in range(steps):
        observation, rew, term, trunc, info  = env.step(np.array([1,1,1]))
        # if crab population drop, catch rate should not be zero 
        if (sum(prev_state) > sum(env.state)):
            assert observation[0] != -1 or observation[1] != -1
        assert rew < 0 # try to discourage laying all traps for each timestep

    assert info == {}
    assert trunc == False
    assert term == False
    assert sum(env.state) < 100000

def test_cpue_2():
    env = greenCrabSimplifiedEnv()
    env.reset()
    max_action = 2000
    
    # check if return 0 when sum(action_natural_units) = 0
    action = np.array([-1,-1, -1])
    action_natural_units = np.maximum( max_action * (1 + action)/2 , 0.) # action_natural_units = 0
    observation = env.observations
    assert all((env.cpue_2(observation,  action_natural_units))== np.float32([0,0]))

    # when sum(action_natural_units) > 0
    action = np.array([1,1, 1])
    action_natural_units = np.maximum( max_action * (1 + action)/2 , 0.)
    observation = env.observations
    cpue_2_value = env.cpue_2(observation,  action_natural_units)
    assert -1<cpue_2_value[0]<1 
    assert -1<cpue_2_value[1]<1 

def test_reset():
    env = greenCrabSimplifiedEnv()
    steps = env.Tmax 
    # run the simulation util the 
    for i in range(steps):
        observation, reward, terminated, truncated, info = env.step(np.array([0,0, 0])) # set constant amount of trap every year

    # if final observation space is [-1, -1], set it to other obseravtion space for test
    if all(observation == np.array([-1,-1])):
        observation = np.array([1,1])

    # reset the obseravtion environment
    new_ob_space, new_info = env.reset()

    assert np.array_equal(new_ob_space, np.array([-1, -1]))

# test reward function for both greenCrab and greenCrabSimplified
def test_reward_func():
    env = greenCrabSimplifiedEnv()
    env.reset()
    # self.state = self.self.init_state()

    # test no trap laid for one timestep when no crab (still potentially negative due to environmental deterioration)
    action = np.array([-1, -1, -1])
    assert env.reward_func(action) <= 0
<<<<<<< HEAD
=======

>>>>>>> f29594d465a8920b847ef7a81497ed057360e7d4
    # test for all trap laid for one timestep when no crab
    action = np.array([1, 1, 1])
    assert env.reward_func(action) < 0 # are we expecting positive when no traps laid when no crabs

    # test no trap when there is a lot of crabs
    env.state = np.array([10., 10., 10., 10., 1000., 10000., 100000., 1000., 100., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.])
    action = np.array([-1, -1, -1])
    assert env.reward_func(action) < 0 # expecting neg reward

    # test all trap laid for one timestep
    action = np.array([1, 1, 1])
    assert env.reward_func(action) < 0


    