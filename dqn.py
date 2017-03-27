from __future__ import print_function
import sys
import time
import gym
import numpy as np

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

from replay import ReplayMemory

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
                 nA,
                 dtype,
                 epsilon):
       
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
         self.dtype                =        dtype
         self.epsilon              =        epsilon


    #def compile(self, optimizer, loss_func):
    def compile(self):
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
        dtype = self.dtype
        Q = self.q_network(self.history_length+1, self.nA).type(dtype)
        target_Q = self.q_network(self.history_length+1, self.nA).type(dtype)
        target_Q.load_state_dict(Q.state_dict())

        return Q, target_Q


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

    def _convert_np_to_torch_variable(self, x, dtype):
        return Variable(torch.from_numpy(x).type(dtype))

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
        last_obs = env.reset()
        idx = 0

        Q, target_Q = self.compile()

        ## test Q network
        # obs_test = np.zeros(shape=(1,4,84,84))
        # obs_test = self._convert_np_to_torch_variable(obs_test, dtype)
        #vals = Q(obs_test)

        optimizer = torch.optim.SGD(Q.parameters(), lr=1e-4, momentum=0.9)
        criterion = torch.nn.SmoothL1Loss()

        num_train_updates = 0
        replay_buffer = self.memory

        train_flag = False
        start_processed = self.preprocessor.process_state_for_memory2(last_obs)
        start_hash = self.memory.hashfunc(start_processed)
        state_hash = start_hash

        while idx < num_iterations:
            idx += 1
            # encoded observations of last_obs: enc_obs
            if idx > self.num_burn_in:
                sample = np.random.sample()
                if sample > self.epsilon:
                    Q_input = self.memory.phi(state_hash)

                    action = Q(Variable(torch.from_numpy(Q_input).type(dtype), volatile=True)).data.max(1)[1]
                    action = action[0,0]
                else:
                    action = np.random.randint(0, self.nA)
            else:
                action = np.random.randint(0, self.nA)

            # take a step
            obs, reward, is_terminal, _ = env.step(action)

            # store info in replay buffer
            st1_hash, st2_hash = replay_buffer.append(last_obs, action, reward, (obs, reward, is_terminal, _))

            if is_terminal:
                obs = env.reset()
                state_hash = start_hash
            else:
                state_hash = st2_hash
            last_obs = obs

            if not train_flag:
                start = time.time()
            ### Train network, only if the experience buffer is already populated
            #  otherwise populate the buffer
            if idx > self.num_burn_in and idx % self.train_freq == 0:
                train_flag = True
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = \
                    replay_buffer.sample(self.batch_size)

                # convert np Tensors to torch variables
                obs_batch = self._convert_np_to_torch_variable(obs_batch, dtype)
                LONG = torch.cuda.LongTensor
                action_batch = self._convert_np_to_torch_variable(action_batch, LONG)
                reward_batch = self._convert_np_to_torch_variable(reward_batch, dtype)
                next_obs_batch = self._convert_np_to_torch_variable(next_obs_batch, dtype)
                not_done_batch = self._convert_np_to_torch_variable(1-done_batch, dtype)

                current_Q_values = Q(obs_batch).gather(1, action_batch.unsqueeze(1))

                # compute best target network value for next state , over all actions
                max_Q_target = target_Q(next_obs_batch).detach().max(1)[0]

                target_Q_values = reward_batch + self.gamma *(not_done_batch * max_Q_target)

                # set gradient to zero before backprop
                optimizer.zero_grad()

                # compute loss using Q and target networks
                loss = criterion(current_Q_values, target_Q_values)

                loss.backward()
                optimizer.step()


                num_train_updates += 1
                curr_time = time.time()
                avg_training_time = (curr_time-start)/num_train_updates

                if num_train_updates % self.target_update_freq == 0:
                    target_Q.load_state_dict(Q.state_dict())

                if num_train_updates % 100 == 0:
                    print('%d: avg training time: %3.5f' %(idx, avg_training_time))

                # Logs



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
    env = gym.make('SpaceInvaders-v0')
    env.reset()

    USE_CUDA = torch.cuda.is_available()
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # parameters
    history_length = 3
    gamma = 0.99
    learning_rate = 1e-4
    epsilon = 0.05
    num_training_samples = int(5e6)
    buffer_size = int(1e6)
    target_update_freq = int(1e4)
    batch_size = 32
    num_burn_in = 100 # 5e5
    train_freq = 1 # 10000
    nA = env.action_space.n

    # create preprocessor class
    preprocessor = AtariPreprocessor(84)
    print('created preprocessor')

    # create replay buffer
    replay_buffer = ReplayMemory(buffer_size, history_length, 84)
    print('created replay buffer')

    # create DQN agent
    agent = DQNAgent(DQN,
                                preprocessor,
                                replay_buffer,
                                policy.GreedyEpsilonPolicy,
                                gamma,
                                target_update_freq,
                                num_burn_in,
                                train_freq,
                                batch_size,
                                history_length,
                                nA,
                                dtype,
                                epsilon)
    print('create DQN agent')

    agent.fit(env, num_training_samples)
