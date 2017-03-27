import numpy as np
import xxhash
from core import ReplayMemory
from preprocessors import AtariPreprocessor,HistoryPreprocessor
import sys

class ReplayMemory(ReplayMemory):
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just randomly draw samples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size, window_length, downsample_img_size):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.max_size=max_size
        self.window_length=window_length
        self.experience=[]
        self.preprocessor = AtariPreprocessor(downsample_img_size)
        self.historytracker = HistoryPreprocessor(window_length)
        self.state_hash_table={}
        self.imgsize=downsample_img_size

    def hashfunc(self,state):
        hashval=xxhash.xxh32(state.tostring()).hexdigest()
        self.state_hash_table[hashval]=state
        return hashval

    def get_state_hash_table_stats(self):
        return len(self.state_hash_table.keys())

    def get_state(self,state_hash):
        if state_hash == 0: return np.zeros((self.imgsize,self.imgsize))
        return self.state_hash_table[state_hash]

    def append(self, state,action,reward,next_tuple):
        if len(self.experience)>self.max_size: self.experience=self.experience[1:]
        st_processed = self.preprocessor.process_state_for_memory2(state)
        st_hash=self.hashfunc(st_processed)
        st1 = next_tuple[0]
        isterminal = next_tuple[2]
        st1_processed = self.preprocessor.process_state_for_memory2(st1)
        st1_hash=self.hashfunc(st1_processed)
        prev_states = self.historytracker.process_state_for_network(st_hash)
        reward = self.preprocessor.process_reward(reward)
        et = (prev_states, st_hash, action, reward, st1_hash, isterminal)
        self.experience.append(et)
        if isterminal: self.end_episode(st1,isterminal)
        return st_hash, st1_hash

    def end_episode(self, final_state, is_terminal):
        self.historytracker.reset()

    def phi(self,state_hash):
        prev_states = self.historytracker.process_state_for_network(state_hash, False)
        processed_states = []
        for s in prev_states:
            processed_states.append(self.get_state(s).astype(np.float32))

        processed_states.append(self.get_state(state_hash).astype(np.float32))

        obs = np.stack(processed_states, axis=0)
        obs = np.stack([obs])
        return obs

    def sample(self, batch_size, indexes=None):
        if indexes: batch_indices=indexes
        else: batch_indices=np.random.choice(len(self.experience), batch_size)
        batch =[]
        for i in batch_indices:
            datapt = self.experience[i]
            action = datapt[2]
            reward = datapt[3]
            is_terminal = datapt[5]

            processed_states = []
            prev_states = datapt[0]
            for s in prev_states:
                processed_states.append(self.get_state(s).astype(np.float32))
            curstate = self.get_state(datapt[1]).astype(np.float32)
            processed_states.append(curstate)
            next_state = self.get_state(datapt[4]).astype(np.float32)
            processed_states.append(next_state)

            obs = np.stack(processed_states[:4], axis=0)
            next_obs = np.stack(processed_states[1:], axis=0)

            batch.append((obs, action, reward, next_obs, is_terminal))

        batch = zip(*batch)
        batch = [np.array(x) for x in batch]

        return batch

    def clear(self):
        raise NotImplementedError('This method should be overridden')
