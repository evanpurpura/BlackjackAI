"""
Evan Purpura
CPSC 481-04
Final Project: Team 16
BLACKJACK AI
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

# different possible states
SCORE = 0
STATE = 0
ACTION = 1
REWARD = 2

EPSILON = 0.1

awards = []


# plays the hand to completion
def play_hand(policy, game):
    results = []
    # start the game
    current_state = game.reset()
    global awards

    while True:
        probability = policy(current_state)

        # get an action based off the probability of the policy
        policy_action = np.random.choice(np.arange(len(probability)), p=probability)

        # print every turn
        print(f"AI: {game.player}:{sum_hand(game.player)} Dealer: {game.dealer}:{sum_hand(game.dealer)} Action: {check_action(policy_action)}")

        # get values from openAI
        next_state, reward, done, _ = game.step(policy_action)
        results.append([current_state, policy_action, reward])

        # once game is over display final hands and reward, then return results
        if done:
            print(f"AI: {sum_hand(game.player)} Dealer: {sum_hand(game.dealer)} Result: {check_reward(reward)}")
            print("------------------------------------------------------------------")
            awards.append(reward)
            return results

        current_state = next_state


# gets the string value of the result
def check_reward(result):
    if result >= 1:
        return "win"
    elif result == 0:
        return "draw"
    elif result == -1:
        return "loss"


# From Blackjack OpenAI
def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


# From Blackjack OpenAI
def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


# gets the string value of the action
def check_action(action):
    if action == 0:
        return "stand"
    else:
        return "hit"


# caculates and prints the percentages of wins, losses, and draws
def get_averages(total_hands):
    wins = 0
    losses = 0
    draws = 0
    global awards

    for result in awards:
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        elif result == -1:
            losses += 1

    wins /= total_hands
    losses /= total_hands
    draws /= total_hands

    wins *= 100
    losses *= 100
    draws *= 100

    print(f"Win rate: {wins}% Loss rate: {losses}% Draw rate: {draws}%")


# monte carlo algorithm: returns the q_estimates
def monte_carlo(game, total_hands):

    total_actions = game.action_space.n
    q_estimate = defaultdict(lambda: np.zeros(total_actions))
    returns = defaultdict(list)

    # returns the new policy based off previous states
    def update_policy(state):

        # finds most optimal action for current state
        optimal_action = np.argmax(q_estimate[state])

        policy = np.ones(total_actions, dtype=float) * EPSILON / total_actions

        policy[optimal_action] = 1 - EPSILON + EPSILON / total_actions

        # return optimal policy
        return policy

    for _ in range(total_hands):
        hand = play_hand(update_policy, game)

        state_action = set([(step[STATE], step[ACTION]) for step in hand])
        for s, a in state_action:

            # find first occurance of state_action
            first_occurance = next(index for index, step in enumerate(hand) if step[STATE] == s and step[ACTION] == a)

            # sum all rewards from first occurance on
            reward_sum = sum([step[REWARD] for step in hand[first_occurance:]])

            returns[(s, a)].append(reward_sum)
            # average the returns
            q_estimate[s][a] = np.mean(returns[(s, a)])

    return q_estimate


# create environment
game = gym.make('Blackjack-v0')
total_hands = 100000
q_estimates = monte_carlo(game, total_hands)

get_averages(total_hands)


# gets value of the max action and assigns it to the state
value_estimates = defaultdict(float)
for state, actions in q_estimates.items():
    value_estimates[state] = np.max(actions)


# creates the grid with possible values of player's hand and dealer's hand
X, Y = np.meshgrid(np.arange(12, 22), np.arange(1, 11))

# gets all the combinations for when the player does not have a usable ace
no_usable_ace = np.apply_along_axis(lambda index: value_estimates[(index[0], index[1], False)], 2, np.dstack([X, Y]))

# gets all the combinations for when the player does have a usable ace
usable_ace = np.apply_along_axis(lambda index: value_estimates[(index[0], index[1], True)], 2, np.dstack([X, Y]))

# creates 3D subplots
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4), subplot_kw={'projection': '3d'})

# creates plot for no_usable aces
ax0.plot_surface(X, Y, no_usable_ace, cmap=plt.cm.PuBuGn_r)
ax0.set_xlabel('Player')
ax0.set_ylabel('Dealer')
ax0.set_zlabel('MC Estimated Value')
ax0.set_title('No Useable Ace')

# creates plot for usable aces
ax1.plot_surface(X, Y, usable_ace, cmap=plt.cm.YlGnBu_r)
ax1.set_xlabel('Player')
ax1.set_ylabel('Dealer')
ax1.set_zlabel('MC Estimated Value')
ax1.set_title('Useable Ace')

# displays plots
plt.show()