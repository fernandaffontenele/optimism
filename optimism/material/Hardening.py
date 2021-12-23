from collections import namedtuple
from optimism.JaxConfig import *

HardeningModel = namedtuple('HardeningModel', ['compute_hardening_energy_density', 'compute_flow_stress'])

def create_hardening_model(properties):
    if properties['hardening model'] == 'linear':
        Y0 = properties['yield strength']
        H = properties['hardening modulus']

        def hardening_energy_density(eqps):
            return linear(eqps, Y0, H)

    elif properties['hardening model'] == 'voce':
        Y0 = properties['yield strength']
        Ysat = properties['saturation strength']
        eps0 = properties['reference plastic strain']

        def hardening_energy_density(eqps):
            return voce(eqps, Y0, Ysat, eps0)

    elif properties['hardening model'] == 'power law':
        Y0 = properties['yield strength']
        n = properties['hardening exponent']
        eps0 = properties['reference plastic strain']

        def hardening_energy_density(eqps):
            return power_law(eqps, Y0, n, eps0)

    else:
        raise valueError('Unknown hardening model specified')

    return HardeningModel(hardening_energy_density, grad(hardening_energy_density))


def linear(eqps, Y0, H):
    return np.where(eqps >= 0.0,
                    Y0*eqps + 0.5*H*eqps**2,
                    np.inf)


def voce(eqps, Y0, Ysat, eps0):
    return np.where(eqps >= 0.0,
                    Ysat*eqps + (Ysat - Y0)*eps0*(np.expm1(-eqps/eps0)),
                    np.inf)


def power_law(eqps, Y0, n, eps0):
    A = n*Y0*eps0/(1.0 + n)
    x = eqps/eps0
    return np.where(x >= 0.0,
                    A*( (1.0 + x)**((n+1)/n) - 1.0 ),
                    np.inf)
