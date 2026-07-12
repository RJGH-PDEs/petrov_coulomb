import numpy as np
import matplotlib.pyplot as plt
import os

save = True
show = False

csv_path = '../time_evolution/conservation/coulomb.csv'
data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
# columns: step, t, mass, px, py, pz, energy
t      = data[:, 1]
mass   = data[:, 2]
px     = data[:, 3]
py     = data[:, 4]
pz     = data[:, 5]
energy = data[:, 6]

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

axes[0].plot(t, mass, color='steelblue', linewidth=1.5)
axes[0].set_ylabel('mass')
axes[0].grid(True, alpha=0.4)
axes[0].ticklabel_format(useOffset=False)

axes[1].plot(t, px, color='tomato',    linewidth=1.5, label='$P_x$')
axes[1].plot(t, py, color='goldenrod', linewidth=1.5, label='$P_y$')
axes[1].plot(t, pz, color='seagreen',  linewidth=1.5, label='$P_z$')
axes[1].axhline(0, color='gray', linewidth=0.7, linestyle='--')
axes[1].set_ylabel('momentum')
axes[1].legend(fontsize=9, loc='right')
axes[1].grid(True, alpha=0.4)
axes[1].ticklabel_format(useOffset=False)

axes[2].plot(t, energy, color='mediumpurple', linewidth=1.5)
axes[2].set_ylabel('energy proxy')
axes[2].set_xlabel('physical time  $t$')
axes[2].grid(True, alpha=0.4)
axes[2].ticklabel_format(useOffset=False)

plt.tight_layout()

os.makedirs('./figures', exist_ok=True)
fig_name = './figures/coulomb_conservation.png'
if show:
    plt.show()
if save:
    fig.savefig(fig_name, dpi=150, bbox_inches='tight')
    print(f'saved {fig_name}')
