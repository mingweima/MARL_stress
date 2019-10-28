from BankSimEnv import BankSimEnv
from MARL.NaiveA2C.ddpg_agent import Agent
import matplotlib.pyplot as plt
import numpy as np


def MA_obs_to_bank_obs(obs, bank):
    bank_obs = obs[bank.BankName]
    # print(f'BANK OBS of {bank.BankName}', bank_obs)
    cash_price, cb_price, gb_price = bank_obs[0]['CASH'], bank_obs[0]['CB'], bank_obs[0]['GB']
    leverage = bank_obs[3]
    return np.asarray([cash_price, cb_price, gb_price, leverage])


agent_dict = {}
env = BankSimEnv()

for idx, name in enumerate(['AT01', 'BE04', 'FR09']):
    agent = Agent(state_size=4, action_size=2, random_seed=idx, name=name)
    agent_dict[name] = agent

for episode in range(100000):
    print(f'=========================================Episode {episode}===============================================')
    current_obs = env.reset()
    play, max_play = 0, 10
    num_default = []
    while play < max_play:
        actions = {}
        for bank_name, bank in env.allAgentBanks.items():
            if bank.DaysInsolvent >= 2:
                continue
            print(f'Round {play}. Bank {bank_name}, CB: {bank.BS.Asset["CB"].Quantity}, GB: {bank.BS.Asset["GB"].Quantity}, CASH: {bank.BS.Asset["CASH"].Quantity}, OTHER: {bank.BS.Asset["OTHER"].Quantity}, LEV: {bank.get_leverage_ratio()}')
            # conversion
            my_obs = MA_obs_to_bank_obs(current_obs, bank)
            current_obs[bank_name] = my_obs
            # choose action
            if bank.DaysInsolvent <= 0:
                action = agent_dict[bank_name].act(current_obs[bank_name], add_noise=False)
            if bank.DaysInsolvent == 1:
                action = [1, 1]
            actions[bank_name] = action  # this is where you use your RLAgents!
        # convert actions
        actions_dict = {}
        for name, action in actions.items():
            action_dict = {}
            action_dict['CB'], action_dict['GB'] = action[0], action[1]
            actions_dict[name] = action_dict
        new_obs, rewards, dones, infos = env.step(actions_dict)
        for bank_name, bank in env.allAgentBanks.items():
            if bank.DaysInsolvent >= 2:
                continue
            my_new_obs = MA_obs_to_bank_obs(new_obs, bank)
            current_obs[bank_name] = my_new_obs
            agent_dict[bank_name].step(current_obs[bank_name], actions[bank_name], rewards[bank_name], my_new_obs, dones[bank_name])
        current_obs = new_obs
        num_default.append(infos['NUM_DEFAULT'])
        play += 1

    # plt.plot(num_default)
    # plt.ylabel('Number of defaults')
    # plt.show()
