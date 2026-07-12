"""
Coefficient evolution for the Coulomb relaxation experiment.
Only l=0 coefficients are non-zero (radial symmetry preserved exactly):
  alpha_{0,0,0}  flat index 0
  alpha_{1,0,0}  flat index 9
  alpha_{2,0,0}  flat index 18
"""

import os, pickle
import numpy as np
import matplotlib.pyplot as plt

SURFACE = '#fcfcfb'
INK2    = '#52514e'
GRID    = '#e1e0d9'
AXIS    = '#c3c2b7'
C0_COL  = '#2a78d6'
C9_COL  = '#1baf7a'
C18_COL = '#eda100'

TAU       = 1e-6
COEFF_DIR = './coeff'

steps = sorted(int(f[:-4]) for f in os.listdir(COEFF_DIR) if f.endswith('.pkl'))
times, c0s, c9s, c18s = [], [], [], []
for s in steps:
    with open(f'{COEFF_DIR}/{s}.pkl', 'rb') as fh:
        c = pickle.load(fh)
    times.append(s * TAU)
    c0s.append(c[0])
    c9s.append(c[9])
    c18s.append(c[18])

times = np.array(times)
c0s   = np.array(c0s)
c9s   = np.array(c9s)
c18s  = np.array(c18s)

fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for spine in ax.spines.values():
    spine.set_color(AXIS)
    spine.set_linewidth(0.8)

mask = times <= 0.1
ax.plot(times[mask], c0s[mask],  color=C0_COL,  lw=2, label=r'$\alpha_{0,0,0}$')
ax.plot(times[mask], c9s[mask],  color=C9_COL,  lw=2, label=r'$\alpha_{1,0,0}$')
ax.plot(times[mask], c18s[mask], color=C18_COL, lw=2, label=r'$\alpha_{2,0,0}$')

ax.axhline(0, color=AXIS, lw=0.8, ls=':', zorder=0)

ax.set_xlabel(r'$t$', color=INK2, fontsize=11)
ax.set_ylabel('coefficient value', color=INK2, fontsize=11)
ax.tick_params(colors=INK2, labelsize=9)
ax.grid(True, which='both', color=GRID, lw=0.6, ls='--', zorder=0)
ax.legend(fontsize=10, framealpha=0.9, edgecolor=AXIS,
          labelcolor=INK2, facecolor=SURFACE,
          loc='center left')

os.makedirs('./figures', exist_ok=True)
out = './figures/coulomb_coeffs.png'
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=SURFACE)
print(f'saved {out}')
plt.show()
