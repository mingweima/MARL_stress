import numpy as np

def MA_obs_to_bank_obs(obs, bank):
    # obs is (bank.AssetMarket.query_price(), bank.BS.Asset, bank.BS.Liability, bank.get_leverage_ratio(), bank.initialBS)
    bank_obs = obs[bank.BankName]
    # print(f'BANK OBS of {bank.BankName}', bank_obs)
    cb_price, gb_price = bank_obs[0]['CB'], bank_obs[0]['GB']
    # print(bank_obs[4].Asset['CB'].Quantity, bank_obs[4].Asset['GB'].Quantity, bank_obs[4].Liability['LOAN'].Quantity)
    cb_left, gb_left, loans_left = bank_obs[1]['CB'].Quantity/(1+bank_obs[4].Asset['CB'].Quantity), bank_obs[1]['GB'].Quantity/(1+bank_obs[4].Asset['GB'].Quantity), bank_obs[2]['LOAN'].Quantity/(1+bank_obs[4].Liability['LOAN'].Quantity)
    leverage = bank_obs[3]
    return np.asarray([cb_price, gb_price, cb_left, gb_left, loans_left, leverage])


def concact_sar(states, actions, rewards, next_states, dones, num_agents=1):
    states_concat, actions_concat, rewards_concat, next_states_concat, dones_concat \
        = np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
    for k in range(num_agents):
        states_concat = np.concatenate((states_concat, states[k]))
        actions_concat = np.concatenate((actions_concat, actions[k]))
        rewards_concat = np.concatenate((rewards_concat, rewards[k]))
        next_states_concat = np.concatenate((next_states_concat, next_states[k]))
        dones_concat = np.concatenate((dones_concat, dones[k]))
    return states_concat, actions_concat, rewards_concat, next_states_concat, dones_concat


def concact_for_agent(dict, num_agents=1):
    concact = np.asarray([])
    for k in range(num_agents):
        concact = np.concatenate((concact, dict[k]))
    return concact