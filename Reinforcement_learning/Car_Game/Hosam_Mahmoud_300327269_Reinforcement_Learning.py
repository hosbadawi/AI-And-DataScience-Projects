import gym
from IPython.display import clear_output
from time import sleep
import numpy as np
import random

def PrintEnvData(Env_Text): # is for printing the number of actions and states of the environments.
    env = gym.make(Env_Text).env
    env.reset()
    print("Action Space {}".format(env.action_space))
    print("State Space {}".format(env.observation_space))

def BrouteForce(Env): # to try all the possible paths for one state.
    epochs = 0
    penalties, reward = 0, 0
    frames = []
    done = False
    
    while not done:
      # automatically selects one random action 
        action = Env.action_space.sample()
        state, reward, done, info = Env.step(action)

        if reward == -10:
            penalties += 1
        
        # Put each rendered frame into dict for animation
        frames.append({
            'frame': Env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
            }
        )
        epochs += 1
        print("Timesteps taken: {}".format(epochs))
        print("Penalties incurred: {}".format(penalties))
    return frames

def print_frames(frames): # to animate the paths.
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        #print(frame['frame'].getvalue())
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)

def Q_Learing_Training_decay(Alpha, Gamma, Epsilon, Env, DecayRate): # Training function using a decay over episodes.
    q_table = np.zeros([Env.observation_space.n, Env.action_space.n])
    global alpha  , gamma, epsilon
    
    # Hyperparameters
    alpha = Alpha
    gamma = Gamma
    epsilon = Epsilon

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    for i in range(1, 100001):
        state = Env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = Env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = Env.step(action) 
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1
            
        if i % 100 == 0:
            clear_output(wait = True)
            print(f"Episode: {i}")
            alpha = abs(alpha - (1/(1 + (DecayRate * 100000))) * alpha)
            # gamma = abs(gamma - (1/(1 + (DecayRate * 100000))) * gamma)  
            # epsilon = abs(epsilon - (1/(1 + (DecayRate * 100000))) * epsilon)
            
            print('Alpha = ', alpha)
            print('Gamma = ', gamma)
            print('Epsilon = ', epsilon)
            
            alpha = Alpha if alpha == 0 else alpha
            # gamma = Gamma if gamma == 0 else gamma
            # epsilon = Epsilon if epsilon == 0 else epsilon    
     
    print("Training finished.\n")
    return q_table

def Q_Learing_Training(Alpha, Gamma, Epsilon, Env): # Training function without parameter tuning.
    q_table = np.zeros([Env.observation_space.n, Env.action_space.n])
    
    # Hyperparameters
    alpha = Alpha
    gamma = Gamma
    epsilon = Epsilon

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    for i in range(1, 100001):
        state = Env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = Env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, info = Env.step(action) 
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1
            
        if i % 100 == 0:
            clear_output(wait = True)
            print(f"Episode: {i}")
            
    print("Training finished.\n")
    return q_table
    
def Evaluation(Q_Table, Env): # to return the performance indicator of each reinforcement model.
    total_epochs, total_penalties, total_reward = 0, 0, 0
    episodes = 1000

    for _ in range(episodes):
        state = Env.reset()
        epochs, penalties, reward = 0, 0, 0
        
        done = False
        while not done:
            action = np.argmax(Q_Table[state])
            state, reward, done, info = Env.step(action)

            if reward == -10:
                penalties += 1           

            epochs += 1
            
        total_reward += reward
        total_penalties += penalties
        total_epochs += epochs
        
        Metric = (total_reward/(total_penalties + total_epochs))

    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")
    print(f"Average reward per episode: {total_reward / episodes}")
    return Metric

def GridSearch(Alphas , Gammas, Epsilons , Env): # to apply GridSearch and optain the optimal hyperparameters.
    Dictionary = {}
    for Alpha in Alphas:
        for Gamma in Gammas:
            for Epsilon in Epsilons:
                q = Q_Learing_Training(Alpha, Gamma , Epsilon, Env)
                Metric = Evaluation(q , Env)
                Dictionary[Metric] = [Alpha, Gamma , Epsilon]
    return Dictionary


#-------------------------------------------------------------------MAIN-------------------------------------------------------------------

env = gym.make("Taxi-v3").env    

PrintEnvData("Taxi-v3")

Alpha_Gamma_Epsilon = [[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]

frames = BrouteForce(env)
print_frames(frames)
BrouteForce(env)

q = Q_Learing_Training(0.2,0.3,0.4,env)
q_decay = Q_Learing_Training_decay(0.2,0.3,0.4,env,0.1)

Evaluation(q , env)
Evaluation(q_decay , env)

Results = GridSearch(Alpha_Gamma_Epsilon[0], Alpha_Gamma_Epsilon[1], Alpha_Gamma_Epsilon[2], env)
Optimal_Hyperparameter = Results[max(Results)]