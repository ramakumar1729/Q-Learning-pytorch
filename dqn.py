from __future__ import print_function
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision.transforms as T
from torch.autograd import Variable

from preprocessors import AtariPreprocessor
from core import ReplayMemory
import policy

"""Main DQN agent."""

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and function parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    history_length: int
    """
    def __init__(self,
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 history_length,
                 nA):
       
         self.q_network            =        q_network                    
         self.preprocessor         =        preprocessor                    
         self.memory               =        memory                    
         self.policy               =        policy                    
         self.gamma                =        gamma                    
         self.target_update_freq   =        target_update_freq                    
         self.num_burn_in          =        num_burn_in                    
         self.train_freq           =        train_freq                    
         self.batch_size           =        batch_size
         self.history_length       =        history_length
         self.nA                   =        nA


    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        Q = self.q_network(self.history_length, self.nA)
        target = self.q_network(self.history_length, self.nA)
        target.load_state_dict(Q.state_dict())

        return


    def select_action(self, state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        pass

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        pass

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        pass

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        pass

class DQN(nn.Module):

    def __init__(self, window=4, nA=18):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(window, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(7*7*64, 512) 
        self.fc2 = nn.Linear(512, nA)

    def init_weights(self):
        self.conv1.weight.data.uniform_(-0.1, 0.1)
        self.conv2.weight.data.uniform_(-0.1, 0.1)
        self.conv3.weight.data.uniform_(-0.1, 0.1)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    env = gym.make('Space-Invaders-0')
    env.reset()

    USE_CUDA = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # parameters
    history_length = 4
    gamma = 0.99
    learning_rate = 1e-4
    epsilon = 0.05
    num_training_samples = 5e6
    buffer_size = 1e6
    target_update_freq = 1e4
    batch_size = 32
    num_burn_in = 5e5
    train_freq = 10000
    nA = env.action_space.n


    # create preprocessor class
    preprocessor = AtariPreprocessor(84)

    # create replay buffer
    replay_buffer = ReplayMemory(buffer_size, history_length)

    # create DQN agent
    agent = DQNAgent().__init__(DQN,
                                preprocessor,
                                replay_buffer,
                                policy.GreedyEpsilonPolicy,
                                gamma,
                                target_update_freq,
                                num_burn_in,
                                train_freq,
                                batch_size,
                                history_length)
