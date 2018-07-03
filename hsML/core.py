# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:22:19 2017

@author: Adam Deller, UNIVERSITY COLLEGE LONDON.
Editted by: Alex Morgan, UNIVERSITY COLLEGE LONDON.
"""
from operator import attrgetter
import attr
import numpy as np
from tqdm import trange
from .drake1999 import quantum_defects
import pandas as pd
from .interaction_matrix import interaction_matrix

#CODATA 2014, DOI: 10.1103/RevModPhys.88.035009
c = 299792458.0 ## speed of light in vacuum
h = 6.626070040e-34
hbar = 1.054571800e-34
Ry = 10973731.568508
e = 1.6021766208e-19
m_e = 9.10938356e-31
alpha = 7.2973525664e-3
m_u = 1.660539040e-27
En_h = alpha**2.0 * m_e * c**2.0
a_0 = hbar/ (m_e * c * alpha)
mu_B = e * hbar / (2.0 * m_e)

## helium
A_r_helium = 4.002603254130
ionization_helium = 1.9831066637e7
mass_helium = A_r_helium * m_u
mass_helium_core = mass_helium - m_e + ionization_helium * h /c
## reduced electron mass/ m_e
mu_me = mass_helium_core / (mass_helium_core + m_e)
## reduced electron mass / core mass,
mu_M = m_e / (mass_helium_core + m_e)
## Rydberg constant for helium
Ry_M = Ry * mu_me
## g-factors
g_L = 1 - m_e / mass_helium_core
g_s = 2.00231930436182

class Basis(object):
    """ Class to represent the a basis of States.
    """
    @attr.s()
    class Params(object):
        """ attrs class to represent the basis parameters.
        """
        n_min = attr.ib(converter=int)
        @n_min.validator
        def check_n_min(self, attribute, value):
            if not value > 0:
                raise ValueError("n_min must be a positive integer.")
                
        n_max = attr.ib(converter=int)
        @n_max.validator
        def check_n_max(self, attribute, value):
            if not value >= self.n_min:
                raise ValueError("n_max must be larger than or equal to n_min.")
  
        S = attr.ib(converter=int)
        @S.validator
        def check_S(self, attribute, value):
            if not value in [0,1]:
                raise ValueError("S must be either 0 or 1.")

        L_max = attr.ib(default=None)
        ML = attr.ib(default=None)        
        ML_max = attr.ib(default=None)
        
    def __init__(self, states, n_min, n_max, S, L_max=None, ML=None, ML_max=None):
        self.states = states
        self.params = self.Params(n_min, n_max, S, L_max, ML, ML_max)

@attr.s()
class State(object):
    """ attrs class to represent the quantum state |n l S J ML>.
    """
    n = attr.ib(convert=int)
    @n.validator
    def check_n(self, attribute, value):
        if not value > 0:
            raise ValueError("n must be a positive integer.")
    L = attr.ib(convert=int)
    @L.validator
    def check_L(self, attribute, value):
        if not value < self.n:
            raise ValueError("L must be an integer smaller than n.")
    S = attr.ib(convert=int)
    @S.validator
    def check_S(self, attribute, value):
        if value not in [0, 1]:
            raise ValueError("S must be an integer 0 or 1.")
        elif self.n == 1 and value != 0:
            raise ValueError("if n == 1, S must be 0.")
    J = attr.ib(convert=int)
    @J.validator
    def check_J(self, attribute, value):
        if self.L == 0 and not value == self.S:
            raise ValueError("If L == 0, J must be equal to S.")
        elif (not self.L - self.S <= value <= self.L + self.S):
            raise ValueError("J must be in range L - S < J < L + S.")
    ML = attr.ib(convert=int)
    @ML.validator
    def check_ML(self, attribute, value):
        if (not -self.L <= value <= self.L):
            raise ValueError("ML must be in the range or -L to L.")

    def __attrs_post_init__(self):
        self.qd = get_qd(self.n, self.L, self.S, self.J)
        self.n_eff = self.n - self.qd
        self.E0 = energy(self.n, self.n_eff)
        
    def __str__(self):
        """ print quantum numbers like |n l S J ML >
        """
        return u"\u2758 {} {} {} {} {} \u27E9".format(self.n, self.L, self.S, self.J, self.ML)
    
    def asdict(self):
        """ quantum numbers as a dictionary.
        """
        return attr.asdict(self)

    def tex(self, show_MJ=True):
        """ Tex string of the form n^{2S + 1}L_{J} (M_J = {MJ})
        """
        L = 'SPDFGHIKLMNOQRTUVWXYZ'[int(self.L%22)]
        tex_str = r'$%d^{%d}'%(self.n, 2*self.S + 1) + L + r'_{%d}'%(self.J)
        if show_MJ:
            tex_str = tex_str + '\,' + r'(M_L = %d)$'%self.ML
        else:
            tex_str = tex_str + r'$'
        return tex_str

class Hamiltonian(object):
    """ The total Hamiltonian matrix.  Each element of the basis set is an
        instance of the class 'State', which represents |n l S J MJ>.
    """
    def __init__(self, n_min, n_max, L_max=None, S=None, ML=None, ML_max=None):
        self.n_min = n_min
        self.n_max = n_max
        self.L_max = L_max
        self.S = S
        self.ML = ML
        self.ML_max = ML_max
        self.basis = basis_states(n_min, n_max, L_max=L_max, S=S, ML=ML, ML_max=ML_max)
        self.sort_basis('E0', inplace=True)
        self.num_states = len(self.basis.states)
        self._h0_matrix = None
        self._stark_matrix = None
        self._zeeman_matrix = None
      
    def sort_basis(self, attribute, inplace=False):
        """ Sort basis on attribute.
        """
        sorted_basis = sorted(self.basis.states, key=attrgetter(attribute))
        if inplace:
            self.basis.states = sorted_basis
        return sorted_basis

    def attrib(self, attribute):
        """ List of given attribute values from all elements in the basis, e.g., J or E0.
        """
        return [getattr(el, attribute) for el in self.basis.states]

    def where(self, attribute, value):
        """ Indexes of where basis.attribute == value.
        """
        arr = self.attrib(attribute)
        return [i for i, x in enumerate(arr) if x == value]

    def h0_matrix(self, **kwargs):
        """ Unperturbed Hamiltonian.
        """
        cache = kwargs.get('cache_matrices', True)
        if self._h0_matrix is None or cache is False:
            self._h0_matrix = np.diag(self.attrib('E0'))
        return self._h0_matrix

    def stark_map(self, Efield, Bfield=0.0, **kwargs):
        """ The eigenvalues of H_0 + H_S + H_Z, for a range of electric fields.
        
            args:
                Efield           dtype: list      units: V / m      

                Bfield=0.0       dtype: float     units: T
            
            kwargs:
                field_angle=0.0    dtype: [float]

                                 specifies the angle between the electric and magnetic fields.
                                 
                eig_vec=False    dtype: bool

                                 returns the eigenvalues and eigenvectors for 
                                 every field value.

                eig_vec_elements=None     dtype: list

                                 calculate the sum of the square of the amplitudes
                                 of the components of the listed basis states for 
                                 each eigenvector, e.g., eig_vec_elements=[1, 3, 5].
                                 Requires eig_vec=False.

            Nb. A large map with eignvectors can take up a LOT of memory.
        """
        if (not kwargs.get('field_angle', 0.0) == 0.0) and \
            ((not self.ML == None) or (not self.ML_max == None)):
            print('WARNING: If the fields are not parallel then all'+\
                  ' ML sub-manifolds are required for accurate results!')
        
        tqdm_kwargs = dict([(x.replace('tqdm_', ''), kwargs[x]) for x in kwargs.keys() if 'tqdm_' in x])
        get_eig_vec = kwargs.get('eig_vec', False)
        get_eig_vec_elements = kwargs.get('eig_vec_elements', None)
        num_fields = len(Efield)
        # initialise output arrays
        eig_val = np.empty((num_fields, self.num_states), dtype=float)
        if get_eig_vec:
            eig_vec = np.empty((num_fields, self.num_states, self.num_states), dtype=float)
        elif get_eig_vec_elements is not None:
            eig_vec_elements = []
        # optional magnetic field
        if Bfield != 0.0:
            Bz = mu_B * Bfield / En_h
            self._zeeman_matrix = interaction_matrix(matrix_type='zeeman', basis=self.basis, **kwargs)
            H_Z = Bz * self._zeeman_matrix.matrix
        else:
            H_Z = 0.0
        # loop over electric field values
        self._stark_matrix = interaction_matrix(matrix_type='stark', basis=self.basis, **kwargs)
        for i in trange(num_fields, desc="diagonalise Hamiltonian", **tqdm_kwargs):
            Fz = Efield[i] * e * a_0 / En_h
            H_S = Fz * self._stark_matrix.matrix / mu_me
            # diagonalise, assuming matrix is Hermitian.          
            if get_eig_vec:
                # eigenvalues and eigenvectors
                eig_val[i], eig_vec[i] = np.linalg.eigh(self.h0_matrix(**kwargs) + H_S + H_Z)
            elif get_eig_vec_elements is not None:
                # eigenvalues and partial eigenvector amplitudes
                eig_val[i], vec = np.linalg.eigh(self.h0_matrix(**kwargs) + H_S + H_Z)
                eig_vec_elements.append(vec[:,get_eig_vec_elements])
            else:
                # eigenvalues
                eig_val[i] = np.linalg.eigh(self.h0_matrix(**kwargs) + H_S + H_Z)[0]
        # output
        if get_eig_vec:
            return eig_val * En_h, eig_vec
        elif get_eig_vec_elements is not None:
            return eig_val * En_h, np.array(eig_vec_elements)
        else:
            return eig_val * En_h       

def get_qd(n, L, S, J, **kwargs):
    """ calculate the quantum defect.
    """
    max_iterations = kwargs.get('max_iterations', 10)
    precision = kwargs.get('precision', 1e-12)
    if L in quantum_defects[S]:
        if J in quantum_defects[S][L]:
            # quantum defect components
            delta = quantum_defects[S][L][J]
            # iteratively calculate quantum defects
            qd_sub1 = delta[0]
            for i in range(max_iterations):
                m = n - qd_sub1
                qd = delta[0]
                for j, d in enumerate(delta[1:]):
                    qd = qd + d * m**(-2*(j + 1))
                # check convergence
                if abs(qd - qd_sub1) < precision:
                    break
                else:
                    qd_sub1 = qd
        else:
            qd = np.nan
    else:
        qd = 0.0
    return qd

def energy(n, n_eff, Z=1):
    """ the ionization energy (atomic units) with relativistic and finite mass corrections.
        Drake 1999 (doi: 10.1238/Physica.Topical.083a00083), eqn. 21
    """
    # TODO - special case for n=1
    en = -0.5 * (1.0 / n_eff**2.0 - \
                 3.0 * alpha**2.0 / (4.0 * n**4.0) + \
                 mu_M**2.0 * ((1 + (5.0/ 6.0) * (Z * alpha)**2.0) / n**2.0))
    return mu_me * en 

def basis_states(n_min, n_max, **kwargs):
    """ Generate the basis set: a list of instances of the attrs class State that 
        satisfy the given ranges of quantum numbers.  By default, all possible 
        states in the range of n_min to n_max are returned.
        
        args:
            n_min             Minimum value of the principal quantum number.

            n_max             Maximum value of the principal quantum number.
        
        kwargs:
            L_max = None      Maximum value of the orbital angular momentum quantum number.
                              If L_max is None 0 < L < n.

            S = None          Value of the total spin quanum number. If S is None S = [0, 1].

            ML = None         Value of the projection of the total angular momentum
                              quantum number. If MJ is None -J <= MJ <= J.

            ML_max = None     Maximum of the absolute value of the projection of the
                              total angular momentum quantum number. If MJ_max and MJ
                              are None -J <= MJ <= J.
    """
    L_max  = kwargs.get('L_max', None)
    S      = kwargs.get('S', None)
    ML     = kwargs.get('ML', None)
    ML_max = kwargs.get('ML_max', None)
    states  = []
    n_rng  = np.arange(n_min, n_max + 1, dtype='int')
    # loop over n range
    for n in n_rng:
        # Don't add if n==1 and S==1
        if not(n==1 and S==1):
            if L_max is not None:
                _L_max = min(L_max, n - 1)
            else:
                _L_max = n - 1
            L_rng = np.arange(0, _L_max + 1, dtype='int')
            # loop over L range
            for L in L_rng:
                if L == 0:
                    J = S
                else:
                    J = L
                # loop over ML range
                if ML is None:
                    for _ML in np.arange(-L, L + 1):
                        if (ML_max is None) or (abs(_ML) <= ML_max):
                            states.append(State(n, L, S, J, _ML))
                elif -L <= ML <= L:
                    states.append(State(n, L, S, J, ML))
    return Basis(states, n_min, n_max, S, L_max, ML, ML_max)

def constants_info():
    constant_vals = {
        'speed of light in vacuum, $c$': c,
        'Planks constant, $h$': h,
        'Reduced Planks constant, $\hbar$': hbar,
        'Rydberg constant, $R_{\infty}$': Ry,
        'electron charge, $e$': e,
        'fine structure constant': alpha,
        'atomic mass': m_u,
        'Hatree energy': En_h,
        'Bohr radius, $a_0$': a_0,
        'Bohr magneton, $\mu_B$': mu_B,
        'ionization energy of helium': ionization_helium,
        'mass of helium': mass_helium,
        'mass of helium (a.u.)': A_r_helium,
        'mass of helium core': mass_helium_core,
        'Reduced electron mass / electron mass': mu_me,
        'Reduced electron mass / core mass': mu_M,
        'Rydberg constant for helium': Ry_M
    }
    df = pd.DataFrame(list(constant_vals.items()), columns=['Constant', 'Value'])
    df['Value'] = df['Value'].map('{:.14g}'.format)
    return df