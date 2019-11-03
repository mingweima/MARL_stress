PROJECT_PATH = '/Users/mmw/Documents/GitHub/MARL_stress/'


class GameParams(object):
    def __init__(self,
                 BANK_BS_PATH,
                 CifuentesImpact_LAMBDA,
                 INITIAL_SHOCK,
                 MARKET_TOTAL_VALUE,
                 ):
        self.BANK_BS_PATH = PROJECT_PATH + BANK_BS_PATH
        self.CifuentesImpact_LAMBDA = CifuentesImpact_LAMBDA
        self.INITIAL_SHOCK = INITIAL_SHOCK
        self.MARKET_TOTAL_VALUE = MARKET_TOTAL_VALUE


TWO_AGENT_SAFE = GameParams(
                 BANK_BS_PATH='Simulator/initial_BS/' + 'two_agent_safe.csv',
                 CifuentesImpact_LAMBDA=0.1,
                 INITIAL_SHOCK=0.0,
                 MARKET_TOTAL_VALUE={'CASH': 1, 'CB': 1e6, 'GB': 1e6, 'OTHER': 1}
                 )

ONE_AGENT_FORCE_SALE = GameParams(
                       BANK_BS_PATH='Simulator/initial_BS/' + 'single_bank_game.csv',
                       CifuentesImpact_LAMBDA=0.1,
                       INITIAL_SHOCK=0.0,
                       MARKET_TOTAL_VALUE={'CASH': 1, 'CB': 1e6, 'GB': 1e6, 'OTHER': 1}
                       )


GAME_PARAMS = TWO_AGENT_SAFE
