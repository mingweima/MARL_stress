import numpy as np


def CifuentesImpact(current_price, fraction_to_sell, l=0.1):
    beta = -10 * np.log(1 - l)
    new_price = current_price * np.exp(-fraction_to_sell * beta)
    return new_price