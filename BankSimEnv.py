from ray.rllib.env import MultiAgentEnv
from AgentBank import Asset, Liability, BalanceSheet, AgentBank
from AssetMarket import AssetMarket
from ImpactFunctions import CifuentesImpact
import matplotlib.pyplot as plt
import numpy as np

from os import path
# params
shock = 0.2
bspath = path.abspath(path.join(path.dirname(__file__), "Bank3.csv"))


def load_bs():
    with open(bspath, 'r') as data:
        bs_from_csv = data.read().strip().split('\n')[1:]
    BalanceSheets = {}
    for bs in bs_from_csv:
        assets, liabilities = {}, {}
        # extract different asset/liability types from the doc
        row = bs.split(' ')
        bank_name, equity, leverage, debt_sec, gov_bonds = row
        equity = float(equity)

        debt_sec = float(debt_sec)
        gov_bonds = eval(gov_bonds)
        corp_bonds = debt_sec - gov_bonds

        asset = equity / (float(leverage) / 100)
        cash = 0.0 * asset # assume all cash has already be used up to pay back leverage
        other_asset = asset - debt_sec - cash

        liability = asset - equity
        loan = liability

        assets['CASH'], assets['CB'], assets['GB'], assets['OTHER'] = \
            Asset('CASH', cash, CifuentesImpact), Asset('CB', corp_bonds, CifuentesImpact), \
            Asset('GB', gov_bonds, CifuentesImpact), Asset('OTHER', other_asset, CifuentesImpact)

        liabilities['LOAN'] = Liability('LOAN', loan)

        BS = BalanceSheet(assets, liabilities)
        BalanceSheets[bank_name] = BS
    return BalanceSheets


def initialize_asset_market():
    assets = {}
    assets['CASH'], assets['CB'], assets['GB'], assets['OTHER'] = \
        Asset('CASH', 1e6, CifuentesImpact), Asset('CB', 1e5, CifuentesImpact), \
        Asset('GB', 3e5, CifuentesImpact), Asset('OTHER', 1e5, CifuentesImpact)
    return AssetMarket(assets)


class BankSimEnv(MultiAgentEnv):
    def __init__(self):
        self.allAgentBanks = {}
        self.initialEquity = {}
        self.DefaultBanks = []  # list of names that has defaulted
        self.AssetMarket = initialize_asset_market()
        self.time = 0

    def reset(self):
        """Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """
        self.allAgentBanks = {}
        self.DefaultBanks = []  # list of names that has defaulted
        self.AssetMarket = initialize_asset_market()
        init_balance_sheets = load_bs()
        for bank_name, BS in init_balance_sheets.items():
            self.allAgentBanks[bank_name] = AgentBank(bank_name, self.AssetMarket, BS)
        self.AssetMarket.apply_initial_shock('GB', shock)
        for bank_name, bank in self.allAgentBanks.items():
            self.initialEquity[bank_name] = bank.get_equity_value()
        obs = {}
        price_dict = self.AssetMarket.query_price()
        for bank_name, bank in self.allAgentBanks.items():
            obs[bank.BankName] = (price_dict, bank.BS.Asset, bank.BS.Liability, bank.get_leverage_ratio())
        return obs

    def step(self, action_dict):
        # action_dict: {AgentName: {TYPE: QTY}}
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.
        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        obs, rewards, dones, infos = {}, {}, {}, {}
        # get new prices that reflect the market impact of all orders
        new_prices = self.AssetMarket.process_orders(self.allAgentBanks, action_dict)
        # print(new_prices)
        name_bank_list = self.allAgentBanks.items()
        for bank_name, bank in name_bank_list:
            # if the bank has defaulted, skip
            if bank_name in self.DefaultBanks:
                continue
            # reflect the asset sale on the bank' BS
            # print(bank_name, bank.get_leverage_ratio())
            action = action_dict[bank_name]
            bank.BS.Liability['LOAN'].Quantity -= self.AssetMarket.convert_to_cash(bank, action)
            bank.BS.sell_action(action)
            # force banks to pay back loans to keep leverage ratio above minimal
            minlev = bank.LeverageMin
            if bank.get_leverage_ratio() > minlev:
                pass
            else:
                bank.DaysInsolvent += 1
            # return obs
            obs[bank.BankName] = (new_prices, bank.BS.Asset, bank.BS.Liability, bank.get_leverage_ratio())
            # return reward
            if bank.DaysInsolvent == 1:
                rewards[bank_name] = -10
                bank.time_of_death = self.time
            else:
                rewards[bank_name] = bank.get_equity_value() / self.initialEquity[bank_name]
            # return dones
            if bank.DaysInsolvent == 2:
                dones[bank_name] = True
                self.DefaultBanks.append(bank_name)
                # print(f'Bank {bank_name} defaults! Leverage is {bank.get_leverage_ratio()}!')
            else:
                dones[bank_name] = False
        infos['ASSET_PRICES'], infos['NUM_DEFAULT'] = new_prices, len(self.DefaultBanks)
        allAgents = self.allAgentBanks.values()
        infos['AVERAGE_LIFESPAN'] = sum(self.time if a.DaysInsolvent == 0 else a.time_of_death for a in allAgents) / len(list(allAgents))
        self.time += 1
        return obs, rewards, dones, infos


if __name__ == '__main__':
    env = BankSimEnv()
    init_obs = env.reset()

    def stupid_action(bank):
        action = {}
        if bank.DaysInsolvent == 0:
            action['CB'], action['GB'] = 0.01, 0.01
        elif bank.DaysInsolvent == 1:
            action['CB'], action['GB'] = 1, 1
        return action

    play, max_play = 0, 5
    num_default = []
    while play < max_play:
        actions = {}
        play += 1
        for bank_name, bank in env.allAgentBanks.items():
            print(
                f'Round {play}. Bank {bank_name}, CB: {int(bank.BS.Asset["CB"].Quantity)}, GB: {int(bank.BS.Asset["GB"].Quantity)}, CASH: {int(bank.BS.Asset["CASH"].Quantity)}, '
                f'EQUITY: {int(bank.get_equity_value())}, ASSET: {int(bank.get_asset_value())}, LIABILITY: {int(bank.get_liability_value())}, LEV: {int(bank.get_leverage_ratio() * 10000)} bps')
            actions[bank_name] = stupid_action(bank)  # this is where you use your RLAgents!
        obs, _, _, infos = env.step(actions)
        num_default.append(infos['NUM_DEFAULT'])

    # plt.plot(num_default)
    # plt.ylabel('Number of defaults')
    # plt.show()
    #





