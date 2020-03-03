import numpy as np
from config import GAME_PARAMS

class CifuentesImpact:
    def __init__(self, l= GAME_PARAMS.CifuentesImpact_LAMBDA):
        self.l = l
    def impact(self, current_price, fraction_to_sell):
        beta = -10 * np.log(1 - self.l)
        new_price = current_price * np.exp(-fraction_to_sell * beta)
        return new_price