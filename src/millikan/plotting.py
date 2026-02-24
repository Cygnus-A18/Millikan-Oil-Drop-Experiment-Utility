from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import os
from .models import DropletData
from .analysis import get_q, get_r, get_n, get_all_q, get_all_r, get_sigma_q, get_all_sigma_q, fit_e_multistart, compute_e_from_lowest_points
from matplotlib.backends.backend_pdf import PdfPages
from typing import TextIO

def plot_discrete_charge(
    data:DropletData, 
    show_ionization_measurements=True, 
    show_mean_q_lines=True,
    show_only_points=False,
    max_size=np.inf, 
    date=None,
    *,
    n_levels = 13,
    base_e_cap = 1.9e-19,         # used for estimating the unit charge from smallest charges
    estimate_method = "mean",       # "mean" or "median"
    rel_tolerance = 0.5,         # points farther than this fraction of unit charge -> unassigned
    out_path = "data/all_charge_measurements"
):

    plt.figure()
    qs = []

    e_points = [
        q
        for trial in data
        for q in get_all_q(trial)
        if q <= 1.9e-19
    ]

    estimated_e = np.mean(e_points)

    if date is not None:
        data = [trial for trial in data if trial.date == date]
    
    if show_ionization_measurements:
        qs = [q for trial in data for q in get_all_q(trial)]
        sigma_qs = [q for trial in data for q in get_all_sigma_q(trial)]
        xs = [r for trial in data for r in get_all_r(trial)]
    else:
        qs = [get_q(trial) for trial in data]
        sigma_qs = [get_sigma_q(trial) for trial in data]
        xs = [get_r(trial) for trial in data]
    
    filtered = [
        (v, l, s)
        for v, l, s in zip(qs, xs, sigma_qs)
        if (v * (10**19)) <= max_size
    ]

    qs, xs, sigma_qs = map(list, zip(*filtered))
    qs_arr = np.asarray(qs, dtype=float)
    xs_arr = np.asarray(xs, dtype=float)
    sigma_qs_arr = np.asarray(sigma_qs, dtype=float)

    if show_only_points:
        plt.errorbar(
            xs,
            qs,
            yerr=sigma_qs,
            fmt='o',
            markersize=4,
            capsize=3
        )
        plt.ylabel("charge of drop (coulombs)")
        plt.xlabel("radius of drop (meters)")
        plt.savefig("data/charge_points.pdf", format="pdf")
        plt.close()
        return

    # estimated_e, _, _ = fit_e_multistart(data)
    estimated_e, _ = compute_e_from_lowest_points(data)

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
        plt.errorbar(
            xs_arr[mask],
            qs_arr[mask],
            yerr=sigma_qs_arr[mask],
            fmt='o',
            markersize=4,
            color=level_colors[k],
            label=f"{k+1}e",
            capsize=3
        )

    if np.any(~assigned):
        plt.errorbar(
            xs_arr[~assigned],
            qs_arr[~assigned],
            yerr=sigma_qs_arr[~assigned],
            fmt='o',
            markersize=4,
            color="black",
            label="unassigned",
            capsize=3
        )

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

    parts = []

    parts.append("with_ionization" if show_ionization_measurements else "without_ionization")
    parts.append("mean_q_lines" if show_mean_q_lines else "expected_e_lines")
    if max_size < np.inf:
        parts.append(f"max_size_{max_size}")

    if date is not None:
        parts.append(str(date))

    suffix = "_".join(parts)
    output_path = os.path.join(f"{out_path}_{suffix}.pdf")
    
    if not show_ionization_measurements:
        out_path = "data/charge_measurements_without_ionizations.pdf"

    plt.ylabel("charge of drop (coulombs)")
    plt.xlabel("radius of drop (meters)")
    plt.savefig(output_path, format="pdf")
    plt.close()

def plot_each_ionization(data, max_size=np.inf, filename="data/ionization_plots.pdf"):
    with PdfPages(filename) as pdf:
        all_xs = [r for trial in data for r in get_all_r(trial)]

        for i, trial in enumerate(data):
            if len(trial.ionized_rise_times) == 0:
                continue

            fig, ax = plt.subplots(figsize=(7, 4))

            qs = get_all_q(trial)
            sigma_qs = get_all_sigma_q(trial)
            xs = get_all_r(trial)

            n = np.round(get_n(trial))
            if n == 0:
                ax.errorbar(xs, qs, yerr=sigma_qs, fmt='o', capsize=5)
                ax.set_title(f"Trial {i}")
                pdf.savefig(fig)
                plt.close(fig)
                continue

            e_est = get_q(trial) / n
            es = e_est * np.arange(1, 11)

            ax.errorbar(xs, qs, yerr=sigma_qs, fmt='o', capsize=5)

            for qe in es:
                if (qe * 1e19) <= max_size:
                    ax.axhline(y=qe, linestyle="--", linewidth=1)
                else:
                    break

            ax.set_title(f"Trial {i}")
            ax.set_xlabel("r (meters)")
            ax.set_ylabel("q (coulombs)")
            ax.set_xlim(min(all_xs) - 0.5e-7,max(all_xs) + 0.5e-7)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

# chat GPT written formatted table output of Data
def generate_table_plaintext(data: DropletData, filename: str) -> None:
    """
    Write a neatly formatted plain-text table of all trials in `data` to `filename`.

    Output includes, for each Trial:
      - date, R
      - for each ionization state (0 = base/un-ionized, 1..N = ionized sets):
          n_rise, mean_rise, sigma_rise, n_fall, mean_fall, sigma_fall
      - plus the raw rise/fall time lists for that state (truncated if long)

    Notes:
      - Uses fixed-width columns so it aligns in any text editor.
      - NaNs are displayed as 'nan'.
    """

    def fmt_num(x: float, width: int = 12, prec: int = 5) -> str:
        # Handle None/NaN cleanly
        if x is None:
            return " " * (width - 4) + "None"
        try:
            if isinstance(x, (float, np.floating)) and np.isnan(x):
                return " " * (width - 3) + "nan"
        except Exception:
            pass
        # Use general formatting but cap precision
        s = f"{float(x):.{prec}g}"
        if len(s) > width:
            # fall back to scientific
            s = f"{float(x):.{max(1, prec - 2)}e}"
        return s.rjust(width)

    def fmt_int(x: int, width: int = 5) -> str:
        return str(int(x)).rjust(width)

    def fmt_text(x: object, width: int) -> str:
        s = "" if x is None else str(x)
        if len(s) > width:
            s = s[: max(0, width - 1)] + "…"
        return s.ljust(width)

    def fmt_list(vals, max_items: int = 10, prec: int = 6) -> str:
        # vals is expected to be list-like
        if vals is None:
            return "[]"
        try:
            vals_list = list(vals)
        except Exception:
            return str(vals)

        if len(vals_list) == 0:
            return "[]"

        shown = vals_list[:max_items]
        parts = []
        for v in shown:
            try:
                if v is None:
                    parts.append("None")
                elif isinstance(v, (float, np.floating)) and np.isnan(v):
                    parts.append("nan")
                else:
                    parts.append(f"{float(v):.{prec}g}")
            except Exception:
                parts.append(str(v))

        suffix = "" if len(vals_list) <= max_items else f", … (+{len(vals_list) - max_items})"
        return "[" + ", ".join(parts) + suffix + "]"

    def write_line(f: TextIO, s: str = "") -> None:
        f.write(s + "\n")

    # Column layout for the summary rows
    # (keep stable widths so it looks good in monospaced fonts)
    col_trial = 5
    col_state = 7
    col_date = 12
    col_r = 10
    col_n = 5
    col_mean = 12
    col_sig = 12

    header = (
        f"{'TRIAL'.ljust(col_trial)} "
        f"{'STATE'.ljust(col_state)} "
        f"{'DATE'.ljust(col_date)} "
        f"{'R'.ljust(col_r)} "
        f"{'nR'.rjust(col_n)} {('rise').rjust(col_mean)} {('σ_rise').rjust(col_sig)} "
        f"{'nF'.rjust(col_n)} {('fall').rjust(col_mean)} {('σ_fall').rjust(col_sig)}"
    )
    rule = "-" * len(header)

    with open(filename, "w", encoding="utf-8") as f:
        # Title / meta
        write_line(f, "DropletData Trials Summary")
        write_line(f, rule)

        if not hasattr(data, "trials") or len(data.trials) == 0:
            write_line(f, "(no trials)")
            return

        write_line(f, header)
        write_line(f, rule)

        # Summary table: one row per trial per ionization state
        for i, trial in enumerate(data.trials):
            # Determine how many states exist for this trial
            # state 0 is base; states 1..N correspond to ionized lists provided
            n_states = max(len(getattr(trial, "all_rise_times", [])), len(getattr(trial, "all_fall_times", [])))
            if n_states == 0:
                n_states = 1

            for state in range(n_states):
                rise_list = trial.all_rise_times[state] if state < len(trial.all_rise_times) else []
                fall_list = trial.all_fall_times[state] if state < len(trial.all_fall_times) else []

                mean_rise = trial.average_rise_times[state] if state < len(trial.average_rise_times) else np.nan
                mean_fall = trial.average_fall_times[state] if state < len(trial.average_fall_times) else np.nan

                sig_rise = trial.sigma_rise_times[state] if state < len(trial.sigma_rise_times) else np.nan
                sig_fall = trial.sigma_fall_times[state] if state < len(trial.sigma_fall_times) else np.nan

                date_str = str(getattr(trial, "date", ""))
                r_val = getattr(trial, "R", "")

                row = (
                    f"{fmt_text(i, col_trial)} "
                    f"{fmt_text(state, col_state)} "
                    f"{fmt_text(date_str, col_date)} "
                    f"{fmt_text(r_val, col_r)} "
                    f"{fmt_int(len(rise_list), col_n)} {fmt_num(mean_rise, col_mean)} {fmt_num(sig_rise, col_sig)} "
                    f"{fmt_int(len(fall_list), col_n)} {fmt_num(mean_fall, col_mean)} {fmt_num(sig_fall, col_sig)}"
                )
                write_line(f, row)

        write_line(f, rule)
        write_line(f)

        # Detailed section: raw lists (truncated) for each trial/state
        write_line(f, "Details (raw lists; truncated)")
        write_line(f, rule)

        for i, trial in enumerate(data.trials):
            date_str = str(getattr(trial, "date", ""))
            r_val = getattr(trial, "R", "")
            write_line(f, f"Trial {i} | date={date_str} | R={r_val}")

            n_states = max(len(getattr(trial, "all_rise_times", [])), len(getattr(trial, "all_fall_times", [])))
            if n_states == 0:
                n_states = 1

            for state in range(n_states):
                rise_list = trial.all_rise_times[state] if state < len(trial.all_rise_times) else []
                fall_list = trial.all_fall_times[state] if state < len(trial.all_fall_times) else []

                mean_rise = trial.average_rise_times[state] if state < len(trial.average_rise_times) else np.nan
                mean_fall = trial.average_fall_times[state] if state < len(trial.average_fall_times) else np.nan
                sig_rise = trial.sigma_rise_times[state] if state < len(trial.sigma_rise_times) else np.nan
                sig_fall = trial.sigma_fall_times[state] if state < len(trial.sigma_fall_times) else np.nan

                label = "base" if state == 0 else f"ionized_{state}"
                write_line(f, f"  State {state} ({label}):")
                write_line(f, f"    rise  n={len(rise_list)}  mean={fmt_num(mean_rise, width=0).strip()}  sigma={fmt_num(sig_rise, width=0).strip()}")
                write_line(f, f"    fall  n={len(fall_list)}  mean={fmt_num(mean_fall, width=0).strip()}  sigma={fmt_num(sig_fall, width=0).strip()}")
                write_line(f, f"    rise_times: {fmt_list(rise_list)}")
                write_line(f, f"    fall_times: {fmt_list(fall_list)}")

            write_line(f)  # blank between trials

def generate_table_latex(data: DropletData, filename: str) -> None:
    """
    Write ONLY a LaTeX table to `filename`.

    The table contains one row per (trial, ionization state), where:
      state 0 = base/un-ionized
      state 1..N = ionized sets

    Columns:
      Trial | State | Date | R | n_rise | rise_mean | sigma_rise | n_fall | fall_mean | sigma_fall
    """

    def tex_escape(s: str) -> str:
        # Minimal escaping for typical fields (dates/labels). Numbers won't need this.
        repl = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        out = []
        for ch in str(s):
            out.append(repl.get(ch, ch))
        return "".join(out)

    def is_nan(x) -> bool:
        try:
            return x is None or (isinstance(x, (float, np.floating)) and np.isnan(x))
        except Exception:
            return x is None

    def fmt_num(x, digits: int = 5) -> str:
        # LaTeX-friendly numeric formatting
        if is_nan(x):
            return r"\mathrm{nan}"
        try:
            return f"{float(x):.{digits}g}"
        except Exception:
            return tex_escape(str(x))

    def fmt_int(x) -> str:
        try:
            return str(int(x))
        except Exception:
            return tex_escape(str(x))

    rows: list[str] = []

    trials = getattr(data, "trials", [])
    for i, trial in enumerate(trials):
        n_states = max(len(getattr(trial, "all_rise_times", [])), len(getattr(trial, "all_fall_times", [])))
        if n_states == 0:
            n_states = 1

        date_str = tex_escape(getattr(trial, "date", ""))
        r_val = getattr(trial, "R", "")
        r_str = tex_escape(r_val) if isinstance(r_val, str) else fmt_num(r_val)

        for state in range(n_states):
            rise_list = trial.all_rise_times[state] if state < len(trial.all_rise_times) else []
            fall_list = trial.all_fall_times[state] if state < len(trial.all_fall_times) else []

            mean_rise = trial.average_rise_times[state] if state < len(trial.average_rise_times) else np.nan
            mean_fall = trial.average_fall_times[state] if state < len(trial.average_fall_times) else np.nan
            sig_rise = trial.sigma_rise_times[state] if state < len(trial.sigma_rise_times) else np.nan
            sig_fall = trial.sigma_fall_times[state] if state < len(trial.sigma_fall_times) else np.nan

            rows.append(
                " & ".join(
                    [
                        fmt_int(i),
                        fmt_int(state),
                        date_str,
                        r_str,
                        fmt_int(len(rise_list)),
                        fmt_num(mean_rise),
                        fmt_num(sig_rise),
                        fmt_int(len(fall_list)),
                        fmt_num(mean_fall),
                        fmt_num(sig_fall),
                    ]
                )
                + r" \\"
            )

    # Only output the table (no extra text)
    latex = "\n".join(
        [
            r"\begin{tabular}{r r l l r r r r r r}",
            r"\hline",
            r"Trial & State & Date & $R$ & $n_R$ & $\langle t_R\rangle$ & $\sigma_R$ & $n_F$ & $\langle t_F\rangle$ & $\sigma_F$ \\",
            r"\hline",
            *rows,
            r"\hline",
            r"\end{tabular}",
            "",
        ]
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex)
