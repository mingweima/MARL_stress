import matplotlib.pyplot as plt
import numpy as np

from environment.BankSimEnv import BankSimEnv
from Naive_MARL.NaiveA2C.ddpg_agent import Agent
from utils.tools import MA_obs_to_bank_obs
from config import GAME_PARAMS

# loop over shocks
eoe_equities = []
impact_ls = [0.001 * x for x in range(0, 200, 5)]
for l in impact_ls:
    agent_dict = {}
    env = BankSimEnv(shock=0.05, Cifuentes_Impact_lambda=l)
    env.reset()

    bank_names = list(env.allAgentBanks.keys())
    print(f'Cifuentes_Impact_lambda = {l}! All {len(bank_names)} participants are: {bank_names}.')
    for idx, name in enumerate(bank_names):
        agent = Agent(state_size=6, action_size=2, random_seed=0, name=name)
        agent_dict[name] = agent

    round_to_print = 50
    average_lifespans = []
    total_equities = []
    for episode in range(GAME_PARAMS.EPISODES):
        if episode == 0 or episode % round_to_print == 0:
            # print(f'=========================================Episode {episode}===============================================')
            a = 1
        current_obs = env.reset()
        play, max_play = 0, 5
        num_default = []
        while play < GAME_PARAMS.MAX_PLAY:
            actions = {}
            for bank_name, bank in env.allAgentBanks.items():
                if bank_name in env.DefaultBanks:
                    continue
                if episode % round_to_print == 0 and False:
                     print(f'Round {play}. Bank {bank_name}, CB: {int(bank.BS.Asset["CB"].Quantity)}, GB: {int(bank.BS.Asset["GB"].Quantity)}',
                        f'EQUITY: {int(bank.get_equity_value())}, ASSET: {int(bank.get_asset_value())}, LIABILITY: {int(bank.get_liability_value())}, LEV: {int(bank.get_leverage_ratio() * 10000)} bps')
                # conversion
                my_obs = MA_obs_to_bank_obs(current_obs, bank)
                current_obs[bank_name] = my_obs
                # choose action
                actions[bank_name] = agent_dict[bank_name].act(current_obs[bank_name].astype(float))
                # print(episode, play, bank_name, actions[bank_name])
            # convert actions
            actions_dict = {}
            for name, action in actions.items():
                action_dict = {}
                action_dict['CB'], action_dict['GB'] = action[0], action[1]
                actions_dict[name] = action_dict
            new_obs, rewards, dones, infos = env.step(actions_dict)
            for bank_name, bank in env.allAgentBanks.items():
                if bank_name in env.DefaultBanks:
                    continue
                my_new_obs = MA_obs_to_bank_obs(new_obs, bank)
                current_obs[bank_name] = my_new_obs
                agent_dict[bank_name].step(current_obs[bank_name], actions[bank_name], rewards[bank_name], my_new_obs, dones[bank_name])
            current_obs = new_obs
            num_default.append(infos['NUM_DEFAULT'])
            play += 1
            if play == max_play:
                # print(infos['AVERAGE_LIFESPAN'])
                average_lifespans.append(infos['AVERAGE_LIFESPAN'])
                total_equities.append(infos['TOTAL_EQUITY'])

    eoe_equity = np.asarray(total_equities).max()
    eoe_equities.append(eoe_equity)
print(eoe_equities)
plt.plot(impact_ls, eoe_equities)

fig, ax = plt.subplots()
ax.plot(impact_ls, eoe_equities)
ax.set(xlabel='initial asset shock', ylabel='end of episode equity',
       title='Relation of shock and equity for Naive A2C, five agents')
# ax.set(xlabel='initial asset shock', ylabel='end of episode equity',
#        title=f'Heuristic Action buffer = {0.045}, five agents')
# ax.grid()
plt.show()





