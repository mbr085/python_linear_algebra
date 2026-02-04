import numpy as np
import sympy as sp

def normer_forste_element(vektor):
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
    forste_ikke_null_indeks = np.argmax(vektor != 0)
    
    # Hent verdien til det første ikke-null elementet
    forste_ikke_null_element = vektor[forste_ikke_null_indeks]
    
    # Returner den normaliserte vektoren
    return vektor / forste_ikke_null_element

def gauss_jordan(matrise, epsilon=1e-8):
    """
    Utfører Gauss-Jordan eliminasjon på en gitt matrise.
    
    Parametere:
        matrise (np.ndarray): En 2D Numpy-array som representerer matrisen.
        epsilon (float): En liten verdi for å håndtere numerisk presisjon.
    
    Returnerer:
        np.ndarray: En rekkeredusert matrise i redusert trappeform.
    """
    # Sett svært små verdier til null for å unngå numeriske feil
    matrise[np.abs(matrise) < epsilon] = 0
    
    # Hvis matrisen kun består av nuller, returner den uendret
    if np.allclose(matrise, 0):
        return matrise
    
    # Hvis matrisen har én rad, normaliser den første radens første ikke-null element
    elif len(matrise) == 1:
        matrise[0] = normer_forste_element(matrise[0])
        return matrise
    
    # Beregn radenes summer for å normalisere matrisen (gjør tallene mer håndterbare)
    rad_maksimummer = np.max(np.abs(matrise), axis=1)
    rad_maksimummer[rad_maksimummer <= epsilon] = 1
    matrise = matrise / rad_maksimummer[:, None]  # Normaliser hver rad
    # Finn kolonner som inneholder ikke-null elementer
    ikke_null_kolonner = np.any(matrise != 0, axis=0)
    
    # Finn indeksen til den første kolonnen med ikke-null elementer
    forste_ikke_null_kolonne_index = np.argmax(np.abs(ikke_null_kolonner))
    forste_ikke_null_kolonne = matrise[:, forste_ikke_null_kolonne_index]
    # Finn indeksen til raden med største verdi i den valgte kolonnen (pivot rad)
    pivot_rad_indeks = np.argmax(forste_ikke_null_kolonne)
    
    # Normaliser pivot-raden
    pivot_rad = normer_forste_element(matrise[pivot_rad_indeks])
    
    # Bytt plass på pivot-raden og den første raden
    matrise[pivot_rad_indeks] = matrise[0]
    matrise[0] = pivot_rad
    # Utfør eliminering for å gjøre alle elementene under pivoten null
    if matrise[0, forste_ikke_null_kolonne_index] > epsilon:
        matrise[1:] -= (matrise[1:, 0] / matrise[0, forste_ikke_null_kolonne_index])[:, None] * matrise[0]

    # Kall Gauss-Jordan rekursivt på den nedre delmatrisen

    matrise[1:, forste_ikke_null_kolonne_index:] = gauss_jordan(matrise[1:, forste_ikke_null_kolonne_index:])
    
    # Gjør den første raden null over pivot-posisjonene til de øvrige radene.
    for rad in matrise[1:]:
        if np.any(rad != 0):  # Hvis raden ikke er null
            # Finn indeksen til det første ikke-null elementet i raden
            forste_ikke_null_kolonne_index = np.argmax(rad != 0)
            
            # Trekk fra et multiplum av denne raden for å gjøre elementet over pivot null
            matrise[0] -= (matrise[0, forste_ikke_null_kolonne_index] / rad[forste_ikke_null_kolonne_index]) * rad
    
    # Returner den resulterende matrisen
    return matrise


def pivot_posisjoner(matrise):
    """
    Finner pivotposisjonene i en rekkeredusert matrise.
    """
    pivot_sett = set()
    pivot_rader = []
    pivot_kolonner = []
    for rad_indeks, rad in enumerate(matrise):
        if np.any(rad != 0):
            første_ikke_null_kolonne = np.argmax(rad != 0)
            pivot_rader.append(rad_indeks)
            pivot_kolonner.append(første_ikke_null_kolonne)
    return pivot_rader, pivot_kolonner

def frie_parametre(matrise):
    """
    Finner bundne og frie parametere i en rekkeredusert matrise.
    """
    _, pivot_kolonner = pivot_posisjoner(matrise)
    alle_kolonner = set(range(matrise.shape[1]))
    return sorted(alle_kolonner.difference(pivot_kolonner))

def null_rom(matrise):
    """
    Finner en basis for nullrommet til en matrise.
    """
    nullrom_basis = []
    redusert_matrise = gauss_jordan(matrise)
    pivot_rader, pivot_kolonner = pivot_posisjoner(redusert_matrise)
    frie = frie_parametre(redusert_matrise)
    
    for fri in frie:
        vektor = np.zeros((matrise.shape[1], 1), dtype=matrise.dtype)
        vektor[fri] = 1
        for rad, kolonne in zip(pivot_rader, pivot_kolonner):
            vektor[kolonne] = -np.sum((redusert_matrise @ vektor)[rad])
        nullrom_basis.append(vektor)
    
    return nullrom_basis


def partikulaer_losning(koeffisientmatrise, høyreside=None):
    """
    Finner en partikulær løsning til ligningssystemet Ax = b.
    """
    if høyreside is None:
        høyreside = np.zeros(koeffisientmatrise.shape[1])
    if len(høyreside.shape) == 1:
        høyreside = høyreside[:, None]
    
    utvidet_matrise = gauss_jordan(np.hstack([koeffisientmatrise, høyreside]))
    radindekser, kolonneindekser = pivot_posisjoner(utvidet_matrise[:, :-1])
    løsning = np.zeros((koeffisientmatrise.shape[1], 1))
    redusert_høyreside = utvidet_matrise[:, -1]
    
    for rad, kolonne in zip(radindekser, kolonneindekser):
        løsning[kolonne] = redusert_høyreside[rad]

    assert np.allclose(koeffisientmatrise @ løsning[None, :], høyreside), "Det finnes ingen løsning til det lineære ligningssystemet"
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

def invers_matrise(A):
    assert A.shape[0] == A.shape[1], "matrisen A skal være kvadratisk"
    B = gauss_jordan(np.hstack([A, np.eye(A.shape[0])]))
    return B[:, A.shape[1]:]
