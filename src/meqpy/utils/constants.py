import scipy.constants as const

ELEMENTARY_CHARGE = const.elementary_charge  # C
ELECTRON_MASS = const.electron_mass  # kg

BOHR = const.physical_constants["Bohr radius"][0] * 1e10  # Å

PLANCK = const.Planck  # J·s
PLANCK_EV = const.Planck / ELEMENTARY_CHARGE  # eV·s
HBAR = const.hbar  # J·s
HBAR_EV = const.hbar / ELEMENTARY_CHARGE  # eV·s

EV_TO_K2 = 2 * ELECTRON_MASS * ELEMENTARY_CHARGE / HBAR**2 * 1e-20  # 1/V/Å²

G0 = ELEMENTARY_CHARGE / PLANCK  # Conductance quantum for each spin channel in 1/V/s
