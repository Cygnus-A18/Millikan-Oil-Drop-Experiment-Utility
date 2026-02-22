import matplotlib.pyplot as plt
import numpy as np
from .models import DropletData
from matplotlib.backends.backend_pdf import PdfPages

def plot_discrete_charge(
    data:DropletData, 
    show_ionization_measurements=False, 
    show_mean_q_lines=False, 
    max_size=np.inf, 
    date=None,
    *,
    n_levels: int = 13,
    base_e_cap: float = 1.9e-19,         # used for estimating the unit charge from smallest charges
    estimate_method: str = "mean",       # "mean" or "median"
    rel_tolerance: float = 0.5,         # points farther than this fraction of unit charge -> unassigned
):

    plt.figure()
    qs = []
    es = np.linspace(1,13,13) * 1.602e-19

    e_points = [
        q
        for trial in data
        for q in trial.get_all_q()
        if q <= 1.9e-19
    ]


    estimated_e = np.mean(e_points)

    if date is not None:
        data = [trial for trial in data if trial.date == date]
    
    if show_ionization_measurements:
        qs = [q for trial in data for q in trial.get_all_q()]
        xs = [r for trial in data for r in trial.get_all_r()]
    else:
        qs = [trial.get_q() for trial in data]
        xs = [trial.get_r() for trial in data]
    
    filter = [(v,l) for v, l in zip(qs, xs) if (v * (10**19)) <= max_size]
    qs, xs = map(list, zip(*filter))
    qs_arr = np.asarray(qs, dtype=float)
    xs_arr = np.asarray(xs, dtype=float)

    small = qs_arr[qs_arr <= base_e_cap]
    if small.size == 0:
        small = qs_arr[qs_arr > 0]

    if small.size == 0:
        estimated_e = 1.602e-19
    else:
        estimated_e = np.median(small) if estimate_method.lower() == "median" else np.mean(small)

    unit_e = estimated_e if show_mean_q_lines else 1.602e-19
    levels = np.arange(1, n_levels + 1, dtype=float) * unit_e

    dists = np.abs(qs_arr[:, None] - levels[None, :])
    nearest_idx = np.argmin(dists, axis=1)
    nearest_dist = dists[np.arange(qs_arr.size), nearest_idx]

    assigned = nearest_dist <= (rel_tolerance * estimated_e)

    n_levels = len(levels)

    base_cmap = plt.get_cmap("tab20").colors

    order = list(range(0, 20, 2)) + list(range(1, 20, 2))
    reordered_colors = [base_cmap[i] for i in order]

    level_colors = reordered_colors[:n_levels]

    for k in range(n_levels):
        mask = assigned & (nearest_idx == k)
        if np.any(mask):
            plt.scatter(xs_arr[mask], qs_arr[mask],
                        s=20,
                        color=level_colors[k],
                        label=f"{k+1}e")

    if np.any(~assigned):
        plt.scatter(xs_arr[~assigned], qs_arr[~assigned],
                    s=20, color="black", label="unassigned")

    if show_mean_q_lines == True:
        for k, qe in enumerate(levels):
            if (qe * (10**19)) <= max_size:
                plt.axhline(y=qe,
                            linestyle='--',
                            color=level_colors[k],
                            linewidth=1.5)
            else:
                break

    es_true = np.arange(1, n_levels + 1) * 1.602e-19
    for qe in es_true:
        if (qe * (10**19)) <= max_size:
            plt.axhline(y=qe, linestyle='-', color='black', linewidth=1)
        else:
            break

    if date is not None:
        plt.title(f"trials from {date}")

    plt.ylabel("charge of drop")
    plt.xlabel("mass of drop")
    plt.show()

def plot_each_ionization(data, max_size=np.inf, filename="ionization_plots.pdf"):
    with PdfPages(filename) as pdf:
        for i, trial in enumerate(data):
            if len(trial.ionized_rise_times) == 0:
                continue

            fig, ax = plt.subplots(figsize=(7, 4))

            qs = trial.get_all_q()
            xs = trial.get_all_r()

            n = np.round(trial.get_n())
            if n == 0:
                ax.scatter(xs, qs)
                ax.set_title(f"Trial {i}")
                pdf.savefig(fig)
                plt.close(fig)
                continue

            e_est = trial.get_q() / n
            es = e_est * np.arange(1, 11)

            ax.scatter(xs, qs)

            for qe in es:
                if (qe * 1e19) <= max_size:
                    ax.axhline(y=qe, linestyle="--", linewidth=1)
                else:
                    break

            ax.set_title(f"Trial {i}")
            ax.set_xlabel("r")
            ax.set_ylabel("q")

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)