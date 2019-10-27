#  A bank that has a balance sheet. It does not have any decision making capability (Brain)


class Asset:
    def __init__(self, type, quantity, impact_function):
        self.Type = type
        self.Quantity = quantity
        self.ImpactFunction = impact_function


class Liability:
    def __init__(self, type, quantity):
        self.Type = type
        self.Quantity = quantity


class BalanceSheet:
    def __init__(self, assets, liabilities):
        # dict = {TYPE: Asset/Liability}
        self.Asset = assets
        self.Liability = liabilities

    def sell_action(self, action):
        # deduce the sold assets from BS
        # action = {ASSET_TYPE: QTY}, ASSET_TYPE must not be 'CASH'; it can only be tradable asset
        for atype, qty in action.items():
            self.Asset[atype] -= qty

    def add_cash(self, cash):
        self.Asset['CASH'] += cash



class AgentBank:
    def __init__(self, bank_name, asset_market, balance_sheet):
        self.BankName = bank_name
        self.AssetMarket = asset_market
        self.BS = balance_sheet
        self.LeverageMin = 0.03
        self.DaysInsolvent = 0

    def get_asset_value(self):
        prices = self.AssetMarket.query_price()
        value = 0
        assets = self.BS.Asset
        for atype, asset in assets.items():
            value += asset.Quantity * prices[atype]
        return value

    def get_liability_value(self):
        value = 0
        liabilities = self.BS.Liability
        for ltype, liability in liabilities.items():
            value += liability.Quantity
        return value

    def get_leverage_ratio(self):
        return (self.get_asset_value() - self.get_liability_value())/ self.get_asset_value()






