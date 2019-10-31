from ray.rllib.env import MultiAgentEnv
from Simulator.AgentBank import Asset, Liability, BalanceSheet, AgentBank
from Simulator.AssetMarket import AssetMarket
from Simulator.ImpactFunctions import CifuentesImpact
from copy import deepcopy
from global_params import BANK_BS_PATH, INITIAL_SHOCK, MARKET_TOTAL_VALUE
from os import path
# params
shock = INITIAL_SHOCK
bspath = BANK_BS_PATH


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
        Asset('CASH', MARKET_TOTAL_VALUE['CASH'], CifuentesImpact), Asset('CB', MARKET_TOTAL_VALUE['CB'], CifuentesImpact), \
        Asset('GB', MARKET_TOTAL_VALUE['GB'], CifuentesImpact), Asset('OTHER', MARKET_TOTAL_VALUE['OTHER'], CifuentesImpact)
    return AssetMarket(assets)


class BankSimEnv(MultiAgentEnv):
    def __init__(self):
        self.allAgentBanks = {}
        self.initialEquity = {}
        self.DefaultBanks = []  # list of names that has defaulted
        self.AssetMarket = initialize_asset_market()
        self.Day = 0

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
            self.initialEquity[bank_name] = deepcopy(bank.get_equity_value())
        obs = {}
        price_dict = self.AssetMarket.query_price()
        for bank_name, bank in self.allAgentBanks.items():
            obs[bank.BankName] = bank.return_obs()
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
        name_bank_list = self.allAgentBanks.items()
        for bank_name, bank in name_bank_list:
            if bank_name in self.DefaultBanks:
                continue
            if bank.IsInsolvent is True:
                # already insolvent banks dont do any actions
                action_dict[bank_name] = {}
            # update action_dict to the real actions for alive banks
            action_dict[bank_name] = bank.day_trade(action_dict[bank_name])
        # pool all orders and send to central clearing
        new_prices = self.AssetMarket.process_orders(self.allAgentBanks, action_dict)
        for bank_name, bank in name_bank_list:
            if bank_name in self.DefaultBanks:
                continue
            if bank.IsInsolvent is False:
                bank.BS.Liability['LOAN'].Quantity -= self.AssetMarket.convert_to_cash(bank, action_dict[bank_name])
                bank.BS.sell_action(action_dict[bank_name])
            # return obs
            obs[bank.BankName] = bank.return_obs()
            # return reward
            if bank.IsInsolvent is True:
                rewards[bank_name] = -10
            else:
                rewards[bank_name] = 10 * (bank.get_equity_value() / self.initialEquity[bank_name] - 0.8)
            # return dones
            if bank.IsInsolvent == True:
                dones[bank_name] = True
                self.DefaultBanks.append(bank_name)
                # print(f'Bank {bank_name} defaults! Leverage is {bank.get_leverage_ratio()}!')
            else:
                dones[bank_name] = False

        infos['ASSET_PRICES'], infos['NUM_DEFAULT'] = new_prices, len(self.DefaultBanks)
        allAgents = self.allAgentBanks.values()

        infos['AVERAGE_LIFESPAN'] = 0
        for bank in allAgents:
            if bank.IsInsolvent:
                infos['AVERAGE_LIFESPAN'] += bank.DeathTime/len(list(allAgents))
            else:
                infos['AVERAGE_LIFESPAN'] += bank.Day/len(list(allAgents))

        self.Day += 1
        return obs, rewards, dones, infos


class CollaborativeBankSimEnv(MultiAgentEnv):
    def __init__(self):
        self.allAgentBanks = {}
        self.initialEquity = {}
        self.DefaultBanks = []  # list of names that has defaulted
        self.AssetMarket = initialize_asset_market()
        self.Day = 0

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
            obs[bank.BankName] = bank.return_obs()
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
        name_bank_list = self.allAgentBanks.items()
        central_reward = 0
        for bank_name, bank in name_bank_list:
            if bank_name in self.DefaultBanks:
                continue
            if bank.IsInsolvent is True:
                # already insolvent banks dont do any actions
                action_dict[bank_name] = {}
            # update action_dict to the real actions for alive banks
            action_dict[bank_name] = bank.day_trade(action_dict[bank_name])
        # pool all orders and send to central clearing
        new_prices = self.AssetMarket.process_orders(self.allAgentBanks, action_dict)
        for bank_name, bank in name_bank_list:
            if bank_name in self.DefaultBanks:
                continue
            if bank.IsInsolvent is False:
                bank.BS.Liability['LOAN'].Quantity -= self.AssetMarket.convert_to_cash(bank, action_dict[bank_name])
                bank.BS.sell_action(action_dict[bank_name])
            # return obs
            obs[bank.BankName] = bank.return_obs()
            # return reward
            if bank.IsInsolvent is True:
                central_reward -= 5
            else:
                central_reward += bank.get_equity_value() / self.initialEquity[bank_name]
            # return dones
            if bank.IsInsolvent == True:
                dones[bank_name] = True
                self.DefaultBanks.append(bank_name)
                # print(f'Bank {bank_name} defaults! Leverage is {bank.get_leverage_ratio()}!')
            else:
                dones[bank_name] = False
        for bank_name, bank in name_bank_list:
            rewards[bank_name] = central_reward

        infos['ASSET_PRICES'], infos['NUM_DEFAULT'] = new_prices, len(self.DefaultBanks)
        allAgents = self.allAgentBanks.values()

        infos['AVERAGE_LIFESPAN'] = 0
        for bank in allAgents:
            if bank.IsInsolvent:
                infos['AVERAGE_LIFESPAN'] += bank.DeathTime/len(list(allAgents))
            else:
                infos['AVERAGE_LIFESPAN'] += bank.Day/len(list(allAgents))

        self.Day += 1
        return obs, rewards, dones, infos




if __name__ == '__main__':
    env = BankSimEnv()
    init_obs = env.reset()

    def stupid_action():
        action = {}
        action['CB'], action['GB'] = 0.01, 0.01
        return action

    play, max_play = 0, 20
    num_default = []
    while play < max_play:
        actions = {}
        play += 1
        for bank_name, bank in env.allAgentBanks.items():
            if bank_name in env.DefaultBanks:
                continue
            print(
                f'Round {play}. Bank {bank_name}, CB: {int(bank.BS.Asset["CB"].Quantity)}, GB: {int(bank.BS.Asset["GB"].Quantity)}, CASH: {int(bank.BS.Asset["CASH"].Quantity)}, '
                f'EQUITY: {int(bank.get_equity_value())}, ASSET: {int(bank.get_asset_value())}, LIABILITY: {int(bank.get_liability_value())}, LEV: {int(bank.get_leverage_ratio() * 10000)} bps')
            actions[bank_name] = stupid_action()  # this is where you use your RLAgents!
        obs, _, _, infos = env.step(actions)
        num_default.append(infos['NUM_DEFAULT'])

    # plt.plot(num_default)
    # plt.ylabel('Number of defaults')
    # plt.show()
    #




