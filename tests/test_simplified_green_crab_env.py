from rl4greencrab import greenCrabSimplifiedEnv
import numpy as np

def test_step_function():
    # not catching any turtle, no trap laid
    env = greenCrabSimplifiedEnv()
    env.reset()
    action = np.array([-1,-1,-1])
    observation, rew, term, trunc, info = step(action)
    assert rew < 0 # check if reward negative when do nothing
    
    
    
    