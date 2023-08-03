import numpy as np
import itertools
import random
import torch

def inv_sherman_morrison(u,A_inv):
    Au = np.dot(A_inv, u)
    A_inv -= np.outer(Au, Au)/(1+np.dot(u.T, Au))
    return A_inv

class ContextualBandit():
    def __init__(self, T, n_arms, n_features, h, noise_std = 1):
        self.T = T 
        self.n_arms = n_arms  # number of arms
        self.n_features = n_features # number of features for each arm
        self.h = h  # average reward function
        self.reset()
    def arms(self):
        return range(self.n_arms)
    def reset(self):
        # generate new features and new rewards
        self.reset_features()
        self.reset_rewards()
    def reset_features(self):
        # generate normalized random N(0,1) features 
        x = np.random.randn(self.T, self.n_arms, self.n_features)
        x /= np.repeat(np.linalg.norm(x, axis=-1, ord=2), self.n_features).reshape(self.T, self.n_arms, self.n_features)
        self.features = x
    def reset_rewards(self):
        # generate rewards for each arm in each round, plus Gaussian noise
        self.rewards = np.array(
            [
                self.h(self.features[t, k]) + self.noise_std*np.random.randn()
                for t, k in itertools.product(range(self.T), range(self.n_arms))
            ]
        ).reshape(self.T, self.n_arms)

        # to be used only to compute regret, NOT by the algorithm itself
        self.best_rewards_oracle = np.max(self.rewards, axis=1)
        self.best_actions_oracle = np.argmax(self.rewards, axis=1)