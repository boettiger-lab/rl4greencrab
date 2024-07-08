from rl4greencrab import greenCrabSimplifiedEnv
import numpy as np


def test_action_units():
    env = greenCrabSimplifiedEnv()
    env.reset()
    action = np.array([-1,-1,-1])
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
        observation, rew, term, trunc, info  = env.step(np.array([1,1, 1]))
        # if crab population drop, catch rate should not be zero 
        if (sum(prev_state) > sum(env.state)):
            assert observation[0] != -1 or observation[1] != -1

    assert info == {}
    assert trunc == False
    assert term == False
    assert sum(env.state) < 100000
