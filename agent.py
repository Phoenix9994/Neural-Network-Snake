#Snake Reinforcement Model | Neural Network
#Will be a great way to teach myself PyTorch and also illustrate as a project under Neural Networks, make edits when done, like try changing dot eats to an apple and make cartoonish
#Name of Project: A.I Snake Model - Utilizes neural networks with the Pytorch package to train an A.I to play the classic game snake. Mention adaptive programming
import torch
import random
import numpy as np
from game import SnakeGameAI, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

#NEED TO IMPLEMENT
#To avoid the self collision we have to add snake parts positions to the game state somehow, in the current model it doesn't know where's the body so it will never learn how to avoid it

"""
Agent
-game
-model 

Training
- state= get_state(game)
action = get_move(state):
        - nodel.predict()
reward, game_over, score =game.play_step (action)
new_state = get_state(game)
- remember
-model.train()
"""

#Can Play around with later
MAX_MEMORY=100_000 #Max memory of 100,000 to store amount of items
BATCH_SIZE= 1000 #
LR= 0.0001

class Agent:
    def __init__(self):
        self.n_games=0
        #Controls Randomness
        self.epsilon=0
        #Discount rate
        self.gamma=0.9 # CAN PLAY AROUND MUST BE SMALLER THAN 1
        #deque is a double ended queue
        self.memory= deque(maxlen=MAX_MEMORY) # popleft()
        self.model=Linear_QNet(19,256,3) #changed 11 to 15
        self.trainer=QTrainer(self.model, lr=LR, gamma=self.gamma)
        

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Body proximity
        body_nearby = [
            game.is_collision(Point(head.x + dx, head.y + dy))
            for dx, dy in [(-20, 0), (20, 0), (0, -20), (0, 20)]
        ]#addeD

        # Snake body positions
        body_positions = [
            (segment.x, segment.y) for segment in game.snake[1:]
        ]
        body_map = [int((head.x + dx, head.y + dy) in body_positions) for dx, dy in [(-20, 0), (20, 0), (0, -20), (0, 20)]]
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]
        # Add body proximity states
        state.extend(body_nearby)  # Append body proximity info to stat
        state.extend(body_map)  # Adds info on immediate body parts nearby
            

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if  MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory)>BATCH_SIZE:
            mini_sample=random.sample(self.memory, BATCH_SIZE) #list of tuples 
        else:
            mini_sample=self.memory

        #zip(*var)  * unpacks the list of tuples or lists, zip groups them together by their index
        # for this in packs all states, actions etc togehter
        states, actions, rewards, next_states, dones= zip(*mini_sample) 
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #Random moves: tradeoff exploration / exploitation
        self.epsilon=80-self.n_games #CAN PLAY AROUND WITH THIS
        final_move=[0,0,0]
        if random.randint(0,200) < self.epsilon:
            move=random.randint(0,2)
            final_move[move]=1
        else: 
            #PYTORCH DEMO
            #Tensor- similar to a multi- dim array
            state0=torch.tensor(state,dtype=torch.float)
            prediction= self.model(state0)
            move=torch.argmax(prediction).item()
            final_move[move]=1
        return final_move

def train(): 
    plot_scores=[]
    plot_mean_scores=[]
    total_score=0
    record=0
    agent=Agent()
    game= SnakeGameAI()
    while True:
        # get old state 
        state_old=agent.get_state(game)
            
        #get move
        final_move= agent.get_action(state_old)

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new=agent.get_state(game)

        #train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #train long memory, plot result
            game.reset()
            agent.n_games +=1
            agent.train_long_memory()

            if score > record:
                record=score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)


    
if __name__== '__main__':
    train()
