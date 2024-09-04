import pygame, sys, time, random
from pygame.surfarray import array3d
from pygame import display

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

BLACK = pygame.Color(0,0,0)
WHITE = pygame.Color(255, 255, 255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)

class SnakeEnv(gym.Env):
    '''
    Action_Space:
        0 = Up
        1 = Down
        2 = Left
        3 = Right

        Time Limit = 1000 Steps
    '''

    metadata = {'render_mode':['human']}

    def __init__(self):
        '''
        Defines the initial game window size
        '''
        self.action_space = spaces.Discrete(4)
        # observation space is optional -> if image than ts not important
        self.frame_size_x = 200
        self.frame_size_y = 200
        self.game_window = pygame.display.set_mode((self.frame_size_x, self.frame_size_y))
        # reset the game
        self.reset()
        # game is after 1000 steps over
        self.STEP_LIMIT = 1000
        self.sleep = 0

        self.observation_shape = (200, 200, 3)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float16)


    def reset(self):
        '''
        Resets the game, along with the default snake size and spawning food.
        NEEDS to return ONLY observation img as an array
        '''
        self.game_window.fill(BLACK)
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [100-10, 50], [100-20, 50]]
        self.food_pos = self.spawn_food()
        self.food_spawn = True

        self.direction = "RIGHT"
        self.action = self.direction

        self.score = 0
        self.steps = 0
        img = array3d(display.get_surface())
        img = np.swapaxes(img, 0, 1)
        return img

    @staticmethod
    def change_direction(action, direction):
        '''
        Changes direction based on action input.
        Checkes to make sure snake can't go back on itself.
        '''
        if action == 0 and direction != 'DOWN':
            direction = 'UP'
        elif action == 1 and direction != 'UP':
            direction = 'DOWN'
        elif action == 2 and direction != 'LEFT':
            direction = 'RIGHT'
        elif action == 3 and direction != 'RIGHT':
            direction = 'LEFT'

        return direction

    @staticmethod
    def move(direction, snake_pos):
        '''
        Updates snake_pos list to reflect direction change.
        '''
        if direction == 'UP':
            snake_pos[1] -= 10
        elif direction == 'DOWN':
            snake_pos[1] += 10
        elif direction == 'LEFT':
            snake_pos[0] -= 10
        elif direction == 'RIGHT':
            snake_pos[0] += 10

        return snake_pos

    def spawn_food(self):
        '''
        Spawns food in a random location on window size.
        '''
        return [random.randrange(1, (self.frame_size_x//10))*10, random.randrange(1, (self.frame_size_y//10))*10]

    def eat(self):
        '''
        Return Boolean indicating if Snake has "eaten" the white food square
        '''
        return self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]

    def step(self, action):
        '''
        What happens when your agent performs the action on the env?
        '''
        scoreholder = self.score
        reward = 0
        self.direction = SnakeEnv.change_direction(action, self.direction)
        self.snake_pos = SnakeEnv.move(self.direction, self.snake_pos)
        self.snake_body.insert(0, list(self.snake_pos))

        reward = self.food_handler()  # reward_handler

        self.update_game_state()

        reward, done = self.game_over(reward)

        img = self.get_image_array_from_game()  # get observations

        info = {'score':self.score}
        self.steps += 1
        time.sleep(self.sleep)

        return img, reward, done, info

    def food_handler(self):
        '''
        Calculates the reward and respawn food if needed.
        '''
        if self.eat():
            self.score += 1
            reward = 1
            self.food_spawn = False
        else:
          self.snake_body.pop()
          reward = 0 

        if not self.food_spawn:
            self.food_pos = self.spawn_food()
        self.food_spawn = True

        return reward

    def get_image_array_from_game(self):
        '''
        Returns an image of the current game as array
        '''
        img = array3d(display.get_surface())
        img = np.swapaxes(img, 0, 1)
        return img

    def update_game_state(self):
        # Drawing the snake
        self.game_window.fill(BLACK)
        for pos in self.snake_body:
            pygame.draw.rect(self.game_window, GREEN, pygame.Rect(pos[0], pos[1], 10, 10))

        # Drawing of food
        pygame.draw.rect(self.game_window, WHITE, pygame.Rect(self.food_pos[0], self.food_pos[1], 10, 10))


    def game_over(self, reward):
        '''
        Checks if the game is over. 
        Returns Reward, done.
        '''
        # TOUCH BOX
        if self.snake_pos[0] < 0 or self.snake_pos[0] > self.frame_size_x-10:
            return -1, True

        elif self.snake_pos[1] < 0 or self.snake_pos[1] > self.frame_size_y-10:
            return -1, True

        # TOUCH BODY
        for block in self.snake_body[1:]:
            if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                return -1, True
    
        # GAME/TIME IS OVER
        if self.steps >= self.STEP_LIMIT:
            return 0, True

        return reward, False

    def render(self, mode='human'):
        if mode == 'human':
            display.update()

    def close(self):
        pass
