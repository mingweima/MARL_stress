from collections import defaultdict


class AssetMarket:
    def __init__(self, assets):
        # specify which assets to clear and total amount of each asset
        self.Assets = assets  # a dictionary { TYPE: Asset(TYPE, TOTAL_QTY, ImpactFunction) }
        self.prices = defaultdict(lambda: 1.0)  # initialize all prices as 1.0
        self.days = 0  # trading days elapsed

    def query_price(self):
        # returns prices of all assets as dict {TYPE: PRICE}
        return dict(self.prices)

    def convert_to_cash(self, action):
        # action = {TYPE: QTY}
        cash = 0
        prices = self.query_price()
        for atype, qty in action.items():
            cash += prices[atype] * qty
        return cash

    def apply_initial_shock(self, asset_to_shock, shock):
        # initialize asset prices such that there is a shock
        for atype, asset in self.Assets.items():
            if atype == asset_to_shock:
                self.prices[atype] = 1.0 * (1 - shock)
            else:
                self.prices[atype] = 1.0

    def process_orders(self, all_order_list):
        # update market prices after processing all incoming orders from agents
        # all_order_list = { 'BANK_NAME': {A_TYPE: QTY}  } }
        current_prices = self.query_price()
        for atype, asset in self.Assets.items():
            qty = 0.
            for bank_name, bank_order_dict in all_order_list.items():
                qty += bank_order_dict[atype]
            fraction_to_sell = qty/asset.Quantity
            new_price = asset.ImpactFunction(current_prices[atype], fraction_to_sell)
            self.prices[atype] = new_price
        return self.query_price()




