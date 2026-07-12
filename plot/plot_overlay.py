import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, '.')
from lc import linear_comb

# ── flags ─────────────────────────────────────────────────────────────────────
# iteration indices to overlay; physical time = step * tau
STEPS = [1, 1000, 10000, 100000, 1000000]
TAU   = 1e-6

IC_LABEL = r'$\alpha_{0,0,0}=1$, $\alpha_{1,0,0}=-0.6$'

save = True
show = False
# ── end flags ─────────────────────────────────────────────────────────────────

# z-axis cut: theta=0 for pz>0, theta=pi for pz<0
N_PTS = 200
pz    = np.linspace(-5, 5, N_PTS)
r     = np.abs(pz)
t_sp  = np.where(pz >= 0, 0.0, np.pi)
p_sp  = np.zeros(N_PTS)

os.makedirs('./figures', exist_ok=True)


def eval_f(coeff):
    return np.array([
        np.exp(-r[i]**2 / 2) * linear_comb(coeff, r[i], t_sp[i], p_sp[i])
        for i in range(N_PTS)
    ])


def load_step(step):
    with open(f'coeff/{step}.pkl', 'rb') as fh:
        return pickle.load(fh)


colors = cm.plasma(np.linspace(0, 0.9, len(STEPS)))
fig, ax = plt.subplots(figsize=(9, 6))

for color, step in zip(colors, STEPS):
    coeff = load_step(step)
    f     = eval_f(coeff)
    ax.plot(pz, f, color=color, label=f't = {step * TAU:.2g}')

ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
ax.set_xlabel(r'$p_z$')
ax.set_ylabel(r'$f(p)$')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.4)
ax.text(0.02, 0.97, f'IC:  {IC_LABEL}', transform=ax.transAxes,
        fontsize=8, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

fig_name = './figures/coulomb_relaxation_overlay.png'
if show:
    plt.show()
if save:
    fig.savefig(fig_name, dpi=150, bbox_inches='tight')
    print('saved', fig_name)
