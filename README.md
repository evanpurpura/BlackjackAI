# BlackjackAI
CPSC 481-04 Final Project: Team 16
Evan Purpura

For this project I used the OpenAI Blackjack environment.  The sum_hand() and usable_ace() functions are taken from the OpenAI Blackjack source code. https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py.

I would give myself a baseline score of 10%. (to play it safe)

The libraries used in the this project are:
  import gym
  import numpy as np
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  from collections import defaultdict
  
In order to run this program all you need are these libraries, the Blackjack.py file, and python 3.7.  

Blackjack.py:
  The program will output the player's hand, dealer's hand, and the player's action for each turn, and will display the final   result for each game.  The program will also output two 3D charts displaying the Q estimate for hands with a usable ace, and   hands without a usable ace.
