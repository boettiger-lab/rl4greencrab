from rl4greencrab import greenCrabSimplifiedEnv
import numpy as np


def test_action_units():
    env = greenCrabSimplifiedEnv()
    env.reset()
    action = np.array([-1,-1,-1])
    natural_units = np.maximum( env.max_action * (1 + action)/2 , 0.)
    assert np.array_equal(natural_units, np.array([0,0,0]))

def test_no_harvest():
    env = greenCrabSimplifiedEnv()
    env.reset()
    
    steps = 3
    for i in range(steps):
        observation, rew, term, trunc, info  = env.step(np.array([-1,-1, -1]))

    assert info == {}
    assert trunc == False
    assert term == False
    assert rew < 0
    assert sum(env.state) > 0


def test_full_harvest():
    env = greenCrabSimplifiedEnv()
    env.reset()
    
    steps = env.Tmax
    for i in range(steps):
        observation, rew, term, trunc, info  = env.step(np.array([1,1, 1]))

    assert info == {}
    assert trunc == False
    assert term == False
    assert sum(env.state) < 100000
