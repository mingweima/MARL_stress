import numpy as np
from global_params import CifuentesImpact_LAMBDA

def CifuentesImpact(current_price, fraction_to_sell, l=CifuentesImpact_LAMBDA):
    beta = -10 * np.log(1 - l)
    new_price = current_price * np.exp(-fraction_to_sell * beta)
    return new_price