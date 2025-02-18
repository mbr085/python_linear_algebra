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

    return løs_homogent_system_av_differensialligninger(
            overgangsmatrise, starttilstand - ligevektsløsning, tverdier) + ligevektsløsning[:, None]

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

def eulers_metode(f, g, t0, tn, x0, y0, n):
  """
  Anvender Eulers metode for å løse differensialligningen x' = f(t, x, y) og y' = g(t, x, y).

  Args:
    f: Funksjonen som representerer x-delen av høyre side av differensialligningen.
    g: Funksjonen som representerer y-delen av høyre side av differensialligningen.
    t0: Startverdien for t.
    tn: Sluttverdien for t.
    x0: Startverdien for x.
    y0: Startverdien for y.
    n: Antall steg.

  Returns:
    En NumPy array som inneholder de approksimerte verdiene for x i hvert steg,
    En NumPy array som inneholder de approksimerte verdiene for y i hvert steg,
  """

  # Lager en array med t-verdier fra t0 til tn med n+1 punkter
  t_values = np.linspace(t0, tn, n + 1)
  # Initialiserer listene for x- og y-verdiene med startverdiene
  x_values = [x0]
  y_values = [y0]
  # Beregner steglengden
  delta_t = t_values[1] - t_values[0]

  # Gjennomfører Eulers metode for hvert steg
  for i in range(n):
    # Beregner neste x-verdi
    x_next = x_values[-1] + delta_t * f(t_values[i], x_values[-1], y_values[-1])
    # Beregner neste y-verdi
    y_next = y_values[-1] + delta_t * g(t_values[i], x_values[-1], y_values[-1])
    # Legger til de nye verdiene i listene
    x_values.append(x_next)
    y_values.append(y_next)

  # Returnerer t-verdiene og de beregnede x- og y-verdiene som NumPy arrays
  return t_values, np.array(x_values), np.array(y_values)
