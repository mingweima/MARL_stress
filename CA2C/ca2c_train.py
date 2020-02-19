import numpy as np

from environment.BankSimEnv import BankSimEnv
from CA2C.ca2c_agent import CA2C_Agent, Centralized_Critic
from config import GAME_PARAMS
from utils.tools import MA_obs_to_bank_obs

agent_dict = {}
env = BankSimEnv()
env.reset()
local_critic, update_critic = Centralized_Critic(6, 2, 1, num_agents=2), Centralized_Critic(6, 2, 1, num_agents=2)

bank_names = list(env.allAgentBanks.keys())
print(f'Game simulations starting! All {len(bank_names)} participants are: {bank_names}.')
for idx, name in enumerate(bank_names):
    agent = CA2C_Agent(6, 2, 1, name=name, critic_local=local_critic, critic_target=update_critic)
    agent_dict[name] = agent


round_to_print = 100
average_lifespans = []
total_equities = []
for episode in range(GAME_PARAMS.EPISODES):
    if episode == 0 or episode % round_to_print == 0:
        print(f'=========================================Episode {episode}===============================================')
    current_obs = env.reset()
    play, max_play = 0, 5
    num_default = []
    while play < GAME_PARAMS.MAX_PLAY:
        actions = {}
        for bank_name, bank in env.allAgentBanks.items():
            if bank_name in env.DefaultBanks:
                my_obs = np.asarray([0, 0, 0, 0, 0, 0])
                current_obs[bank_name] = my_obs
            if episode % round_to_print == 0:
                 print(f'Round {play}. Bank {bank_name}, CB: {int(bank.BS.Asset["CB"].Quantity)}, GB: {int(bank.BS.Asset["GB"].Quantity)}',
                    f'EQUITY: {int(bank.get_equity_value())}, ASSET: {int(bank.get_asset_value())}, LIABILITY: {int(bank.get_liability_value())}, LEV: {int(bank.get_leverage_ratio() * 10000)} bps')
            if not bank_name in env.DefaultBanks:
                # conversion
                my_obs = MA_obs_to_bank_obs(current_obs, bank)
                current_obs[bank_name] = my_obs
        # choose action
        for bank_name, bank in env.allAgentBanks.items():
            actions[bank_name] = agent_dict[bank_name].act(current_obs)
            # print(episode, play, bank_name, actions[bank_name])
        # convert actions
        actions_dict = {}
        for name, action in actions.items():
            action_dict = {}
            action_dict['CB'], action_dict['GB'] = action[0], action[1]
            actions_dict[name] = action_dict
        new_obs, rewards, dones, infos = env.step(actions_dict)
        new_obs_dict = {}
        for bank_name, bank in env.allAgentBanks.items():
            if bank_name in env.DefaultBanks:
                new_obs_dict[bank_name] = np.asarray([0, 0, 0, 0, 0, 0])
                continue
            new_obs_dict[bank_name] = MA_obs_to_bank_obs(new_obs, bank)
            current_obs[bank_name] = new_obs_dict[bank_name]
        for bank_name, bank in env.allAgentBanks.items():
            if bank_name in env.DefaultBanks:
                continue
            agent_dict[bank_name].step(current_obs, actions, rewards, new_obs_dict, dones)
        current_obs = new_obs
        num_default.append(infos['NUM_DEFAULT'])
        play += 1
        if play == max_play:
            # print(infos['AVERAGE_LIFESPAN'])
            average_lifespans.append(infos['AVERAGE_LIFESPAN'])
            total_equities.append(infos['TOTAL_EQUITY'])