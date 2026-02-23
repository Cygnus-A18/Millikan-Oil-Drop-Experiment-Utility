import numpy as np
from .models import Trial
from scipy.interpolate import UnivariateSpline

def get_weighted_average(values):
        if len(values) == 0:
            return np.nan
        values = np.array(values)
        median = np.median(values)
        rlist = values - median
        MAD = np.median(abs(rlist))
        if MAD == 0:
            return np.mean(values)
        weights = 1 / (1 + (rlist / MAD)**2)

        return np.average(values, weights=weights)

def get_weighted_error(values):
    if len(values) == 0:
        return np.nan

    values = np.array(values)
    median = np.median(values)
    rlist = values - median
    MAD = np.median(np.abs(rlist))

    if MAD == 0:
        return np.std(values, ddof=1) / np.sqrt(len(values))

    weights = 1 / (1 + (rlist / MAD)**2)

    weighted_mean = np.average(values, weights=weights)

    numerator = np.sum(weights**2 * (values - weighted_mean)**2)
    denominator = (np.sum(weights))**2

    variance = numerator / denominator

    return np.sqrt(variance)

def compute_e_from_all_points(data):
    es = []
    for trial in data:
        for i in range(len(trial.all_rise_times)):
            q = get_q(trial, i)
            n = np.round(get_n(trial, i))

            e_val = q / n

            es.append(e_val)
    
    return np.mean(es)

def compute_e_from_lowest_points(data):
    qs = [q for trial in data for q in get_all_q(trial)]
    base_e_cap = 1.9e-19
    qs_arr = np.asarray(qs)
    small = qs_arr[qs_arr <= base_e_cap]
    if small.size == 0:
        small = qs_arr[qs_arr > 0]

    if small.size == 0:
        estimated_e = 1.602e-19
    else:
        estimated_e = np.mean(small)

    return estimated_e

def refine_e(q, e0, iters=1000):
    e = e0
    for _ in range(iters):
        n = np.rint(q / e).astype(int)
        n[n < 1] = 1
        e_new = (n @ q) / (n @ n)
        if abs(e_new - e) / e < 1e-12:
            break
        e = e_new
    return e, n

def fit_e_multistart(data, max_n=10):
    qs = [q for trial in data for q in get_all_q(trial)]
    inits = []
    for qi in qs:
        for k in range(1, max_n + 1):
            inits.append(qi / k)

    best = None
    for e0 in inits:
        e, n = refine_e(qs, e0)
        if n.max() > max_n:
            continue
        resid = qs - n * e
        mse = np.mean(resid**2)
        cand = (mse, e, n)
        if best is None or cand[0] < best[0]:
            best = cand

    if best is None:
        raise RuntimeError("No fit found under the max_n constraint.")
    return best[1], best[2]

def calculate_eta(temp):
    m = 4.7058823529e-8
    b = 1.72941176471e-5
    return m * temp + b

def get_t_from_r(resistance):
    data = np.array([
        (10, 3.239),
        (11, 3.118),
        (12, 3.004),
        (13, 2.897),
        (14, 2.795),
        (15, 2.700),
        (16, 2.610),
        (17, 2.526),
        (18, 2.446),
        (19, 2.371),
        (20, 2.300),
        (21, 2.233),
        (22, 2.169),
        (23, 2.110),
        (24, 2.053),
        (25, 2.000),
        (26, 1.950),
        (27, 1.902),
        (28, 1.857),
        (29, 1.815),
        (30, 1.774),
        (31, 1.736),
        (32, 1.700),
        (33, 1.666),
        (34, 1.634),
        (35, 1.603),
        (36, 1.574),
        (37, 1.547),
        (38, 1.521),
        (39, 1.496),
    ])

    # Sort data by resistance (R) to ensure it is strictly increasing for the spline
    data = data[data[:, 1].argsort()]

    T = data[:, 0]  # Temperature (°C)
    R = data[:, 1]  # Resistance (MΩ)

    spline = UnivariateSpline(R, T, s=0, k=3)
    return spline(resistance)

def compute_amqn(trial,
        ionization_index = 0,
        b = 8.20e-3,
        p = 1.013e5,
        g = 9.8,
        rho = 886,
        # d = 9.17e-3,
        d=7.60e-3,
        x = 0.5e-3,
        V = 500,
        e = 1.602e-19,
    ):
        eta = calculate_eta(get_t_from_r(trial.R))
        t0 = trial.average_fall_times[ionization_index]
        sigma_t0 = trial.sigma_fall_times[ionization_index]
        tplus = trial.average_rise_times[ionization_index]
        sigma_tplus = trial.sigma_rise_times[ionization_index]

        vf = x / t0
        sigma_vf = (x / (t0 ** 2)) * sigma_t0
        vr = x / tplus
        sigma_vr = (x / (tplus ** 2)) * sigma_tplus
        

        a = np.sqrt(((b/(2*p)) ** 2) + (9*eta*vf / (2 * g * rho))) - (b/(2*p))
        m = (4/3) * np.pi * a**3 * rho
        q = (m * g * d * (vf + vr)) / (V * vf)
        n = q / e

        C = (m * g * d) / V

        dq_dvf = -C * (vr / (vf**2))
        dq_dvr = C / vf

        sigma_q = np.sqrt(
            (dq_dvf**2) * (sigma_vf**2) +
            (dq_dvr**2) * (sigma_vr**2)
        )

        return a, m, q, n, sigma_q

def get_q(trial: Trial, ionization_index=0):
    _, _, q, _, _ = compute_amqn(trial, ionization_index=ionization_index)
    return q

def get_a(trial: Trial, ionization_index=0):
    a, _, _, _, _ = compute_amqn(trial, ionization_index=ionization_index)
    return a


def get_m(trial: Trial, ionization_index=0):
    _, m, _, _, _ = compute_amqn(trial, ionization_index=ionization_index)
    return m

def get_n(trial: Trial, ionization_index=0, e_custom=None):
    if e_custom is None:
        _, _, _, n, _ = compute_amqn(trial, ionization_index=ionization_index)
    else:
        _, _, _, n, _ = compute_amqn(
            trial, ionization_index=ionization_index, e=e_custom
        )
    return n

def get_sigma_q(trial: Trial, ionization_index=0):
    _, _, _, _, sigma_q = compute_amqn(trial, ionization_index=ionization_index)
    return sigma_q

def get_r(trial: Trial, ionization_index=0):
    m = get_m(trial, ionization_index=ionization_index)
    rho = 886

    return (3 * m / (4 * np.pi * rho)) ** (1 / 3)

def get_all_q(trial:Trial):
    qs = []
    for i in range(len(trial.all_rise_times)):
        qs.append(get_q(trial, i))

    return qs

def get_all_sigma_q(trial: Trial):
    sigma_qs = []
    for i in range(len(trial.all_rise_times)):
        sigma_qs.append(get_sigma_q(trial, i))

    return sigma_qs

def get_all_r(trial:Trial):
    rs = []
    for i in range(len(trial.all_rise_times)):
        rs.append(get_r(trial, i))

    return rs

def get_all_n(trial:Trial):
    ns = []
    for i in range(len(trial.all_rise_times)):
        ns.append(get_n(trial, i))

    return ns