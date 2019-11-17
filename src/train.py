"""
The train file has to coordinate the REINFORCE algorithm in the main function
"""
import numpy as np
import tensorflow as tf
import cnn
import controller
from child_manager import get_reward

#
def train():
    #with tf.name_scope("train"):
    num_episodes = 100
    num_steps = 10
    max_layers = 2
    controller1 = controller(max_layers)
    for ep in num_episodes:
        cnn1 = cnn()
        initial_state = cnn.state
        rewards = []
        logits = []
        for step in num_steps:
            action, logit = controller1.get_action(initial_state) # what state?
            new_state = cnn1.build_child_arc(action)
            reward = get_reward(new_state)
            logits.append(logit)
            rewards.append(reward)
        controller.update_policy(rewards,logits)    
        
    return


if __name__ == '__main__':
  train()

