import numpy as np
from collections import namedtuple
from functools import singledispatch

## --- Materials ---
class Material():

    def __init__(self, symbol, structure, a, color='black'):

        self.symbol = symbol # str, Chemical symbol
        self.structure = structure
        self.a = a  # Lattice parameter, angström

        self.color = color


    def __str__(self):
        return self.symbol
    def __repr__(self):
        return self.symbol


def distance_reticulaire(hkl, a):
    ''' Distance reticulaire pour système cubique
    '''
    h, k, l = hkl
    return a / np.sqrt(h**2 + k**2 + l**2)


# Generation des hlk
all_hkl = []
for l in range(1, 9):  #  <-- indice max.
    for k in range(l+1):
        for h in range(k+1):
            all_hkl.append(sorted([h, k, l], reverse=True))

all_hkl = sorted(all_hkl, key=lambda x: sum(i**2 for i in x))


# Condition d'existences:
def existence_FCC(h, k, l):
    # même parité
    return h % 2 == k % 2 and k % 2 == l % 2


def existence_BCC(h, k, l):
    # somme paire
    return (h+k+l) % 2 == 0


def existence_ZincBlende(h, k, l):
    # même parité
    meme_parite = (h % 2 == k % 2 and k % 2 == l % 2)
    motif = ((h+k+l-2) % 4 != 0)
    return meme_parite and motif


# tests
assert existence_FCC(1, 1, 1) == True
assert existence_FCC(1, 1, 0) == False
assert existence_FCC(2, 2, 0) == True
assert existence_FCC(2, 1, 0) == False

structure_filters = {'bcc': existence_BCC,
                     'fcc': existence_FCC,
                     'ZincBlende': existence_ZincBlende,
                     'cubic': lambda *x: True}


# --- Bragg Law ---

def deuxtheta(hkl, a, lmbda, order=1):
    ''' Return 2theta angle for the given
        crystallographic plane (hkl) and diffraction order `n`
        `a` is the lattice parameter (cubic)
    '''
    d = distance_reticulaire(hkl, a)
    nlambda_sur_2d = order * lmbda / 2 / d

    if nlambda_sur_2d > 1:
        return np.NaN
    else:
        return 2*np.arcsin(nlambda_sur_2d) * 180/np.pi


@singledispatch
def theoretical_peaks(material, lmbda, deuxtheta_max=145):
    ''' List possible hkl for the stucture ('bcc', 'fcc' or 'Zincblende')
        and corresponding 2theta angle using lattice parameter `a`
        and wavelength `lmbda`

        deuxtheta_max: degree
    '''
    condition_existence = structure_filters[material.structure]
    hkl_list = [{'hkl': hkl,
                 'deuxtheta_theo': deuxtheta(hkl, material.a, lmbda),
                 'material':material}
                for hkl in all_hkl
                if condition_existence(*hkl)]

    # Filter with deuxtheta_max
    hkl_list = [peak for peak in hkl_list
                        if np.isfinite(peak['deuxtheta_theo'])
                        and peak['deuxtheta_theo'] < deuxtheta_max]

    return hkl_list


@theoretical_peaks.register(list)
def _(materials, *args, **kargs):
    ''' Generalisation for multiple materials '''
    return [line for mat in materials
            for line in theoretical_peaks(mat, *args, **kargs)]
