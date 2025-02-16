from .matrix_reduction import finn_egenvektorer_og_egenverdier
from .matrix_reduction import invers_matrise
from .matrix_reduction import partikulaer_losning
import numpy as np

def løs_inhomogent_system_av_differensialligninger(overgangsmatrise, starttilstand, påtrykk, tverdier):
    '''
    Løser et homogent lineært system av differensialligninger analytisk
    
    Parametre:
        overgangsmatrise (np.ndarray): En kvadratisk Numpy-array med koeffisientene for systemet
        starttilstand (np.ndarray): En 1D Numpy-array av samme lengde som overgangsmatrise
        tverdier (np.ndarray): En 1D Numpy-array med verdier av t der løsningen skal beregnes
    
    Retunerer:
        y (np.ndarray): Verdiene av løsningen til sistemet i de angitte tverdier
    '''
    påtrykk = np.array(påtrykk).ravel()
    starttilstand = np.array(starttilstand).ravel()
    ligevektsløsning = partikulaer_losning(overgangsmatrise, påtrykk).ravel()
    starttilstand = starttilstand - ligevektsløsning

    return løs_homogent_system_av_differensialligninger(overgangsmatrise, starttilstand, tverdier) + ligevektsløsning[:, None]

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
    starttilstand = np.array(starttilstand).ravel()
    tverdier = np.array(tverdier).ravel()
    assert overgangsmatrise.shape[0] == overgangsmatrise.shape[1], "overgangsmatrise må være kvadratisk"
    assert overgangsmatrise.shape[0] == starttilstand.shape[0], "overgangsmatrise og starttilstand må ha samme dimensjon"
    assert len(tverdier) > 0, "tverdier må ha minst en verdi"

    egenvektorer_og_egenverdier = finn_egenvektorer_og_egenverdier(overgangsmatrise)
    egenverdier = np.array([[x[0]] for x in egenvektorer_og_egenverdier])
    P = np.hstack([x[2][0] for x in egenvektorer_og_egenverdier])
    return P @ ((invers_matrise(P) @ starttilstand[:, None]) * np.exp(tverdier[None, :] * egenverdier))
