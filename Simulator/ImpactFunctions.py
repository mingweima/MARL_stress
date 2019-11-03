import numpy as np
from global_params import GAME_PARAMS

def CifuentesImpact(current_price, fraction_to_sell, l=GAME_PARAMS.CifuentesImpact_LAMBDA):
    beta = -10 * np.log(1 - l)
    new_price = current_price * np.exp(-fraction_to_sell * beta)
    return new_price