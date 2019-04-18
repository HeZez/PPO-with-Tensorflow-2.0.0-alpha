from mlagents.envs import UnityEnvironment
import numpy as np

train_mode=True
env = UnityEnvironment(file_name=None)

# Set the default brain to work with
default_brain = env.brain_names[0]
brain = env.brains[default_brain]


env_info = env.reset(train_mode=train_mode)[default_brain]

for episode in range(1000):
    
    
    env_info = env.reset(train_mode=train_mode)[default_brain]
    done = False
    
    episode_rewards = 0
    while not done:
        action_size = brain.vector_action_space_size
        if brain.vector_action_space_type == 'continuous':
            act = np.random.randn(len(env_info.agents), action_size[0])
            env_info = env.step(act)[default_brain]
        else:
            action = np.column_stack([np.random.randint(0, action_size[i], size=(len(env_info.agents))) for i in range(len(action_size))])
            env_info = env.step(action)[default_brain]
        episode_rewards += env_info.rewards[0]
        done = env_info.local_done[0]
    print("Total reward this episode: {}".format(episode_rewards))

env.close()   