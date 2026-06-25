import scipy.constants as const

elementary_charge = const.elementary_charge  # C
electron_mass = const.electron_mass  # kg

bohr = const.physical_constants["Bohr radius"][0] * 1e10  # Å

planck = const.Planck  # J·s
planck_ev = const.Planck / elementary_charge  # eV·s
hbar = const.hbar  # J·s
hbar_ev = const.hbar / elementary_charge  # eV·s

ev_to_k2 = 2 * electron_mass / hbar**2 * elementary_charge * 1e-20  # 1/V/Å²

G0 = elementary_charge / planck  # Conductance quantum for each spin channel in 1/V/s
