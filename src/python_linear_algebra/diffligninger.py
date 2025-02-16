from .matrix_reduction import finn_egenvektorer_og_egenverdier
from .matrix_reduction import invers_matrise
import numpy as np

def løs_homogent_system_av_differensialligninger(overgangsmatrise, starttilstand, tverdier):
    '''
    Løser et homogent lineært system av differensialligninger analytisk
    
    Parametre:
        overgangsmatrise (np.ndarray): En kvadratisk Numpy-array med koeffisientene for systemet
        starttilstand (np.ndarray): En 1D Numpy-array av samme lengde som overgangsmatrise
        tverdier (np.ndarray): En 1D Numpy-array med verdier av t der løsningen skal beregnes
    
    Retunerer:
        y (np.ndarray): Verdiene av løsningen til sistemet i de angitte tverdier
    '''
    egenvektorer_og_egenverdier = finn_egenvektorer_og_egenverdier(overgangsmatrise)
    egenverdier = np.array([[x[0]] for x in egenvektorer_og_egenverdier])
    P = np.hstack([x[2][0] for x in egenvektorer_og_egenverdier])
    return P @ ((invers_matrise(P) @ starttilstand[:, None]) * np.exp(tverdier[None, :] * egenverdier))
