import numpy as np
import sympy as sp

def normer_største_element(vektor):
    """
    Normaliserer en vektor ved å dele alle elementer på det første ikke-null elementet.
    
    Parametere:
        vektor (np.ndarray): En 1D Numpy-array som skal normaliseres.
    
    Returnerer:
        np.ndarray: En normalisert vektor der det første ikke-null elementet er 1.
    """
    if np.allclose(vektor, 0):
        return vektor
    # Finn indeksen til det første elementet i vektoren som ikke er null
    største_ikke_null_indeks = np.argmax(np.abs(vektor))
    
    # Hent verdien til det første ikke-null elementet
    største_ikke_null_element = np.abs(vektor[største_ikke_null_indeks])
    
    # Returner den normaliserte vektoren
    return vektor / største_ikke_null_element

def gauss_jordan(matrise, epsilon=1e-8):
    """
    Utfører Gauss-Jordan eliminasjon på en gitt matrise.
    
    Parametere:
        matrise (np.ndarray): En 2D Numpy-array som representerer matrisen.
        epsilon (float): En liten verdi for å håndtere numerisk presisjon.
    
    Returnerer:
        np.ndarray: En rekkeredusert matrise i redusert trappeform.
    """
    normaliser = not np.issubdtype(matrise.dtype, np.integer)
    # Sett svært små verdier til null for å unngå numeriske feil
    matrise[np.abs(matrise) < epsilon] = 0
    
    # Hvis matrisen kun består av nuller, returner den uendret
    if np.allclose(matrise, 0):
        return matrise
    
    # Hvis matrisen har én rad, normaliser den første radens første ikke-null element
    elif len(matrise) == 1:
        if normaliser:
            matrise[0] = normer_største_element(matrise[0])
        else:
            matrise[0] = matrise[0] // np.gcd.reduce(matrise[0])
            største_ikke_null_kolonne = np.argmax(np.abs(matrise[0]))
            matrise[0] = matrise[0] // np.sign(matrise[0, største_ikke_null_kolonne])
        return matrise
    
    # Beregn radenes summer for å normalisere matrisen (gjør tallene mer håndterbare)
    rad_maksimummer = np.max(np.abs(matrise), axis=1)
    rad_maksimummer[rad_maksimummer <= epsilon] = 0
    if normaliser:
        matrise = matrise / rad_maksimummer[:, None]  # Normaliser hver rad
    else:
        matrise = rad_maksimummer[:, None] * matrise - matrise
    # Finn kolonner som inneholder ikke-null elementer
    ikke_null_kolonner = np.flatnonzero(np.any(matrise != 0, axis=0))
    
    # Finn indeksen til den første kolonnen med ikke-null elementer
    forste_ikke_null_kolonne_index = ikke_null_kolonner[np.argmax(np.max(np.abs(matrise[:, ikke_null_kolonner]), axis=0))]
    forste_ikke_null_kolonne = matrise[:, forste_ikke_null_kolonne_index]
    # Finn indeksen til raden med største verdi i den valgte kolonnen (pivot rad)
    pivot_rad_indeks = np.argmax(np.abs(forste_ikke_null_kolonne))

    
    # Normaliser pivot-raden
    pivot_rad = matrise[pivot_rad_indeks].copy()
    if normaliser:
        pivot_rad = normer_største_element(pivot_rad)
    
    # Bytt plass på pivot-raden og den første raden
    matrise[pivot_rad_indeks] = matrise[0]
    matrise[0] = pivot_rad
    # Utfør eliminering for å gjøre alle elementene under pivoten null
    if np.abs(matrise[0, forste_ikke_null_kolonne_index]) > epsilon:
        if normaliser:
            matrise[1:] -= (matrise[1:, forste_ikke_null_kolonne_index] / matrise[0, forste_ikke_null_kolonne_index])[:, None] * matrise[0]
        else:
            matrise[1:] = (matrise[0, forste_ikke_null_kolonne_index] * matrise[1:]) - matrise[1:, forste_ikke_null_kolonne_index][:, None] * matrise[0] 

    # Kall Gauss-Jordan rekursivt på den nedre delmatrisen

    resterende_kolonner = np.concatenate((np.arange(0, forste_ikke_null_kolonne_index), np.arange(forste_ikke_null_kolonne_index + 1, matrise.shape[1])))

    matrise[1:, resterende_kolonner] = gauss_jordan(matrise[1:, resterende_kolonner], epsilon=epsilon)
    
    # Gjør den første raden null over pivot-posisjonene til de øvrige radene.
    matrise_pivot_posisjoner = pivot_posisjoner(matrise[1:])
    if normaliser:
        for rad_idx, col_idx in zip(*pivot_posisjoner(matrise[1:])):
            rad = matrise[1 + rad_idx]
            matrise[0] -= (matrise[0, col_idx] / rad[col_idx]) * rad 
    else:
        for rad_idx, col_idx in zip(*pivot_posisjoner(matrise[1:])):
            rad = matrise[1 + rad_idx]
            matrise[0] = matrise[0] * rad[col_idx] - matrise[0, col_idx] * rad 
            if np.issubdtype(matrise.dtype, np.integer):
                matrise[0] = matrise[0] // np.gcd.reduce(matrise[0])
        største_ikke_null_kolonne = np.argmax(np.abs(matrise[0]))
        matrise[0] = matrise[0] // np.sign(matrise[0, største_ikke_null_kolonne])

    # Bytt rader slik at raden med ikke-null element lengst til venstre kommer først
    mask = np.any(matrise != 0, axis=1)
    maskert_matrise = matrise[mask]
    matrise[mask] = [maskert_matrise[:, np.argmax(maskert_matrise, axis=0)]]

    return matrise


def pivot_posisjoner(matrise):
    """
    Finner pivotposisjonene i en rekkeredusert matrise.
    """
    pivot_rader = []
    pivot_kolonner = []
    for rad_indeks, rad in enumerate(matrise):
        if np.any(rad != 0):
            største_ikke_null_kolonne = np.argmax(np.abs(rad))
            pivot_rader.append(rad_indeks)
            pivot_kolonner.append(største_ikke_null_kolonne)
    return pivot_rader, pivot_kolonner

def frie_parametre(matrise):
    """
    Finner bundne og frie parametere i en rekkeredusert matrise.
    """
    _, pivot_kolonner = pivot_posisjoner(matrise)
    alle_kolonner = set(range(matrise.shape[1]))
    return sorted(alle_kolonner.difference(pivot_kolonner))

def null_rom(matrise, epsilon=1e-8,):
    """
    Finner en basis for nullrommet til en matrise.
    """
    normaliser = not np.issubdtype(matrise.dtype, np.integer)
    nullrom_basis = []
    redusert_matrise = gauss_jordan(matrise, epsilon=epsilon)
    pivot_rader, pivot_kolonner = pivot_posisjoner(redusert_matrise)
    frie = frie_parametre(redusert_matrise)
    
    for fri in frie:
        vektor = np.zeros((matrise.shape[1], 1), dtype=matrise.dtype)
        vektor[fri] = 1
        if normaliser:
            for rad, kolonne in zip(pivot_rader, pivot_kolonner):
                vektor[kolonne] = -np.sum((redusert_matrise @ vektor)[rad]) / redusert_matrise[rad, kolonne]
        else:
            for rad, kolonne in zip(pivot_rader, pivot_kolonner):
                vektor = redusert_matrise[rad, kolonne] * vektor
                vektor[kolonne] = -np.sum((redusert_matrise @ vektor)[rad]) // redusert_matrise[rad, kolonne]
                vektor = vektor // np.gcd.reduce(vektor)
        nullrom_basis.append(vektor)
    
    return nullrom_basis

def partikulaer_losning(koeffisientmatrise, høyreside=None, epsilon=1e-8):
    """
    Finner en partikulær løsning til ligningssystemet Ax = b.
    """
    if høyreside is None:
        høyreside = np.zeros(koeffisientmatrise.shape[1])
    if len(høyreside.shape) == 1:
        høyreside = høyreside[:, None]

    utvidet_null_rom = null_rom(np.hstack([koeffisientmatrise, høyreside]), epsilon=epsilon)
    utvidet_null_rom = [v for v in utvidet_null_rom if np.abs(v[-1]) > epsilon]
    if len(utvidet_null_rom) == 0:
        print("Det finnes ingen løsning til det lineære ligningssystemet")
    
    # if np.issubdtype(koeffisientmatrise.dtype, np.integer):
    #     utvidet_null_rom = [v for v in utvidet_null_rom if v[-1] == 1 or v[-1] == -1]
    #     if len(utvidet_null_rom) == 0:
    #         print("Det finnes ingen heltallsløsning til det lineære ligningssystemet")
    #         v = np.zeros((koeffisientmatrise.shape[1] + 1, 1), dtype=koeffisientmatrise.dtype)
    #         v[-1] = 1
    #     else:
    #         v = utvidet_null_rom[0]
    #     løsning = -v[: -1] // v[-1]
    # else:
    if len(utvidet_null_rom) == 0:
        v = np.zeros((koeffisientmatrise.shape[1] + 1, 1), dtype=koeffisientmatrise.dtype)
        v[-1] = 1
    else:
        v = utvidet_null_rom[0]
    løsning = -v[: -1] / v[-1]
    
    # if not np.allclose(koeffisientmatrise @ løsning[None, :], høyreside):
    #     print("Det finnes ingen løsning til det lineære ligningssystemet")
    return løsning

def finn_egenvektorer_og_egenverdier(A, epsilon=1e-12):
    assert A.shape[0] == A.shape[1], "matrisen A skal være kvadratisk"

    t = sp.symbols('t')  # Definerer symbolet t, som brukes i det karakteristiske polynomet
    B = A - t * sp.eye(A.shape[0])  # Lager matrisen B = A - t*I, hvor I er identitetsmatrisen
    karakteristisk_polynom = B.det()  # Finner determinant av B, som gir det karakteristiske polynomet
    
    egenverdier = sp.solve(karakteristisk_polynom)  # Løser det karakteristiske polynomet for å finne egenverdiene

    # Lager en liste med tupler av evenverdier, multiplisiteter og egenvektorer
    res = []
    # for ev in sorted(egenverdier, key=lambda x: -np.abs(x)):
    for ev in egenverdier:
        ev = complex(ev)
        if np.abs(ev.real) < epsilon:
            ev = ev.imag * 1j
        if np.abs(ev.imag) < epsilon:
            ev = ev.real
        if np.isrealobj(ev) or np.iscomplexobj(A):
            egenvektorer = null_rom((A - ev * np.eye(A.shape[0])))
            res.append((ev, len(egenvektorer), egenvektorer))
    
    return res  # Returnerer en liste med tupler av evenverdier, multiplisiteter og egenvektorer

def complex_to_string(c, precision=7):
    if np.all(c.imag == 0):
        return np.array2string(c.real, precision=precision)
    else:
        return np.array2string(c, precision=precision)

def skriv_ut_egenvektorer_og_multiplikasjon_med_matrise(A, egenverdier_og_egenvektorer, presisjon=3):
    print('Alle vektorer her skal leses som kolonnevektorer\n')
    for key, _, val in sorted(
        egenverdier_og_egenvektorer, 
        key = lambda x: (complex(x[0]).real)**2 + (complex(x[0]).imag)**2):
        print('egenverdi:     ', 
              key)
        for v in val:
            print('egenvektor:    ', complex_to_string(np.array(v, dtype='complex128').ravel(), precision=presisjon))
            print('A @ evenvektor:', complex_to_string(np.array(A @ v, dtype='complex128').ravel(), precision=presisjon))
        print()

def skriv_ut_numpy_egenvektorer_og_multiplikasjon_med_matrise(A, presisjon=3):
    print('Alle vektorer her skal leses som kolonnevektorer\n')
    eigenverdier, eigenvektorer = np.linalg.eig(A)
    for key, val in sorted(zip(eigenverdier, eigenvektorer.T),
                           key = lambda x: (complex(x[0]).real)**2 + (complex(x[0]).imag)**2):
        v = val[:, None]
        print('egenverdi:     ', 
              key)
        print('egenvektor:    ', complex_to_string(np.array(v, dtype='complex128').ravel(), precision=presisjon))
        print('A @ evenvektor:', complex_to_string(np.array(A @ v, dtype='complex128').ravel(), precision=presisjon))
        print()

def invers_matrise(A, normaliser=True, epsilon=1e-8):
    assert A.shape[0] == A.shape[1], "matrisen A skal være kvadratisk"
    B = gauss_jordan(np.hstack([A, np.eye(A.shape[0])]), normaliser=normaliser, epsilon=epsilon)
    return B[:, A.shape[1]:]
