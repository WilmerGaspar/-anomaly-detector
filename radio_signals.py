"""Señales de radio: biblioteca de firmas físicas y detectores con nulo explícito.

Firmas (todas generadas por su ecuación, sin datos memorizados):
- Púlsar: tren de pulsos periódico.
- Ráfaga dispersada (tipo FRB / pulso único de púlsar): retraso por dispersión del plasma
  t(f) = K_DM * DM * f^-2, con K_DM = 4.148808 ms GHz^2 / (pc cm^-3).
- Portadora estrecha con deriva Doppler (tipo búsqueda SETI de banda estrecha).
- Interferencias terrestres (RFI): impulso de banda ancha SIN dispersión (DM = 0) y
  portadora estrecha SIN deriva (0 Hz/s, transmisor fijo respecto a la antena).

Detectores:
- periodicity_search: espectro de potencia normalizado por su mediana móvil y sumas de
  armónicos (1-16). Probabilidad de falsa alarma con Bonferroni sobre frecuencias y sumas.
- dispersion_search: desdispersión incoherente en una rejilla de DM + filtro de caja.
- drift_search: desplazar y sumar en frecuencia (de-Doppler) con la resolución real.
  En ambos: FAP analítica con Bonferroni (ruido gaussiano) Y superar los máximos de 19
  nulos con desplazamientos circulares al azar (rompen la curva f^-2 o la deriva). La
  segunda condición protege si el ruido real tiene colas pesadas.

Una detección dice "esta firma está en los datos por encima de su nulo". La causa
(astrofísica, instrumental o terrestre) no la decide la estadística: por eso DM ≈ 0 y
deriva ≈ 0 se marcan como probable RFI.
"""
from __future__ import annotations

import numpy as np

K_DM_S = 4.148808e-3          # s GHz^2 / (pc cm^-3)
FAP = 0.01                    # probabilidad de falsa alarma global para declarar detección


# ---------------------------------------------------------------- biblioteca de firmas

def gen_pulsar(n=2 ** 16, dt=1e-3, period=0.0893, duty=0.03, amp=0.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    phase = (t / period) % 1.0
    w = duty / 2.355
    pulse = np.exp(-0.5 * (np.minimum(phase, 1 - phase) / w) ** 2)
    return rng.standard_normal(n) + amp * pulse / pulse.std()


def gen_dispersed_burst(nchan=64, ntime=2048, f_lo=1.2, f_hi=1.5, dt=1e-3, dm=300.0, t0=0.3, width=2e-3,
                        amp=1.0, seed=0):
    """Espectro dinámico (canales x tiempo). f en GHz, de f_hi (fila 0) a f_lo."""
    rng = np.random.default_rng(seed)
    f = np.linspace(f_hi, f_lo, nchan)
    t = np.arange(ntime) * dt
    delay = K_DM_S * dm * (f ** -2 - f_hi ** -2)
    dyn = rng.standard_normal((nchan, ntime))
    dyn += amp * np.exp(-0.5 * ((t[None, :] - t0 - delay[:, None]) / width) ** 2)
    return dyn, f


def gen_drifting_tone(nchan=1024, ntime=64, df=3.0, dt=10.0, drift=0.2, f_start_chan=300, amp=1.0, seed=0):
    """Espectrograma (tiempo x canales de frecuencia, df en Hz, dt en s, deriva en Hz/s)."""
    rng = np.random.default_rng(seed)
    dyn = rng.standard_normal((ntime, nchan))
    for i in range(ntime):
        c = f_start_chan + drift * i * dt / df
        j = int(round(c))
        if 0 <= j < nchan:
            dyn[i, j] += amp * 3.0
    return dyn


def gen_rfi_impulse(nchan=64, ntime=2048, t0=0.5, dt=1e-3, amp=1.0, seed=0):
    rng = np.random.default_rng(seed)
    dyn = rng.standard_normal((nchan, ntime))
    dyn[:, int(t0 / dt)] += amp * 3.0
    return dyn


LIBRARY = {
    "pulsar": {"name": "Púlsar (tren de pulsos periódico)", "data": "serie temporal",
               "physics": "Estrella de neutrones en rotación; un pulso por vuelta."},
    "dispersed_burst": {"name": "Ráfaga dispersada (tipo FRB)", "data": "espectro dinámico (canales × tiempo)",
                        "physics": "El plasma retrasa las frecuencias bajas: t ∝ DM·f⁻². DM > 0 indica que la señal "
                                   "atravesó plasma (fuera de la Tierra)."},
    "drifting_tone": {"name": "Portadora estrecha con deriva", "data": "espectrograma (tiempo × canales)",
                      "physics": "Un tono de pocos Hz cuya frecuencia deriva por efecto Doppler (aceleración "
                                 "relativa emisor-antena). Es lo que buscan los programas SETI de banda estrecha."},
    "rfi_impulse": {"name": "RFI: impulso de banda ancha sin dispersión", "data": "espectro dinámico",
                    "physics": "Llega a todas las frecuencias a la vez (DM = 0): origen terrestre casi seguro."},
    "rfi_tone": {"name": "RFI: portadora fija sin deriva", "data": "espectrograma",
                 "physics": "Deriva 0 Hz/s: el emisor no se mueve respecto a la antena (transmisor cercano)."},
}

PUBLIC_CATALOGS = [
    ("ATNF Pulsar Catalogue (psrcat)", "https://www.atnf.csiro.au/research/pulsar/psrcat/",
     "Períodos (P0) y medidas de dispersión (DM) de púlsares conocidos. Exporta CSV y súbelo para comparar."),
    ("CHIME/FRB Catalog", "https://www.chime-frb.ca/catalog", "Ráfagas rápidas de radio con su DM."),
    ("Breakthrough Listen Open Data", "http://seti.berkeley.edu/opendata", "Datos de radio de búsquedas SETI."),
]


# ---------------------------------------------------------------- 1. periodicidad

def _running_median_logblocks(p, first=20, growth=1.1, max_block=400):
    """Mediana del espectro en bloques que crecen con la frecuencia, interpolada en log.
    Sigue el ruido rojo (potencia que cae como f^-a) que una ventana fija no puede seguir:
    medido, con ventana fija de 201 bins el ruido rojo daba 30/30 falsas alarmas.
    (Con ventana fija y relleno 'nearest', el borde de Nyquist daba 48 % de falsas alarmas.)"""
    n = len(p)
    centers, meds = [], []
    i, w = 0, float(first)
    while i < n:
        j = min(n, i + max(int(w), 3))
        centers.append(0.5 * (i + j - 1))
        meds.append(np.median(p[i:j]))
        i = j
        w = min(w * growth, max_block)
    idx = np.arange(n)
    return np.exp(np.interp(np.log1p(idx), np.log1p(centers), np.log(np.maximum(meds, 1e-300))))


def _fold_chi2(x, dt, period, bins=32):
    ph = ((np.arange(len(x)) * dt) / period * bins).astype(int) % bins
    cnt = np.bincount(ph, minlength=bins)
    mean = np.bincount(ph, weights=x, minlength=bins) / np.maximum(cnt, 1)
    return float(np.sum(cnt * mean ** 2))


def _refine_period(x, dt, f0, T, n_grid=201):
    """Afina el periodo plegando la serie en +-1 bin de frecuencia alrededor de f0 (chi^2 del
    perfil). Incertidumbre: anchura a media altura del pico de chi^2 / 2 (al menos el paso)."""
    fs = f0 + np.linspace(-1.0, 1.0, n_grid) / T
    fs = fs[fs > 0]
    chi = np.array([_fold_chi2(x, dt, 1.0 / f) for f in fs])
    i = int(np.argmax(chi))
    half = chi[i] - 0.5 * (chi[i] - np.median(chi))
    above = np.nonzero(chi >= half)[0]
    fwhm = (fs[above.max()] - fs[above.min()]) if above.size > 1 else (fs[1] - fs[0])
    f_best = fs[i]
    return 1.0 / f_best, float(max(fwhm / 2, fs[1] - fs[0]) / f_best ** 2)


def periodicity_search(x, dt, harmonics=(1, 2, 4, 8, 16), f_min=None, min_cycles=20):
    """Espectro de potencia de la serie, normalizado para que el ruido siga Exp(1), y sumas
    incoherentes de armónicos. FAP global con Bonferroni sobre frecuencias x sumas."""
    from scipy.stats import gamma
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 64:
        return {"detected": False, "note": "Serie demasiado corta (< 64 muestras)."}
    x = x - x.mean()
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(n, dt)
    p, f = p[1:-1], f[1:-1]                         # sin DC ni Nyquist (no siguen Exp(1))
    med = _running_median_logblocks(p)
    pn = p / (med / np.log(2))                      # mediana de Exp(1) = ln 2
    # Al menos `min_cycles` periodos dentro de la serie: con ruido rojo las falsas alarmas
    # salian todas en el primer bin (1 ciclo en toda la serie; medido 10/30).
    f_min = max(f_min or 0.0, min_cycles / (n * dt))
    best = {"logsf": 0.0}
    n_trials = 0
    for h in harmonics:
        m = len(pn) // h
        if m < 2:
            continue
        s = np.zeros(m)
        for k in range(1, h + 1):
            s += pn[k - 1:k * m:k][:m]                 # pn[k*(i+1)-1] = potencia en k*f0 (indices absolutos)
        ok = f[:m] >= f_min                            # la restriccion va sobre la fundamental, despues de sumar
        if not ok.any():
            continue
        logsf = np.where(ok, gamma.logsf(s, a=h), 0.0)  # suma de h Exp(1) ~ Gamma(h, 1)
        i = int(np.argmin(logsf))
        n_trials += int(ok.sum())
        if logsf[i] < best["logsf"]:
            best = {"logsf": float(logsf[i]), "harmonics": h, "freq_hz": float(f[i]), "power_sum": float(s[i])}
    p_single = float(np.exp(best["logsf"]))
    fap = float(min(1.0, p_single * n_trials))
    out = {"n_samples": n, "dt_s": dt, "n_trials": int(n_trials), "fap": fap, "detected": fap < FAP,
           "max_period_s": 1.0 / f_min,
           "method": "espectro normalizado por mediana en bloques logarítmicos + sumas de armónicos; FAP = p·N (Bonferroni)"}
    if "freq_hz" in best:
        period, period_err = _refine_period(x, dt, best["freq_hz"], n * dt)
        phase = ((np.arange(n) * dt) / period) % 1.0
        bins = 64
        prof = np.bincount((phase * bins).astype(int), weights=x, minlength=bins)[:bins]
        cnt = np.bincount((phase * bins).astype(int), minlength=bins)[:bins]
        out.update({"best_freq_hz": 1.0 / period, "best_period_s": period, "period_err_s": period_err,
                    "harmonics_summed": best["harmonics"],
                    "folded_profile": (prof / np.maximum(cnt, 1)).tolist(),
                    "spectrum_f": f[::max(1, len(f) // 4000)].tolist(), "spectrum_p": pn[::max(1, len(f) // 4000)].tolist()})
    return out


# ---------------------------------------------------------------- 2. dispersión

def _dedisperse(dyn, delays_samples):
    nt = dyn.shape[1]
    idx = (np.arange(nt)[None, :] + np.asarray(delays_samples, dtype=int)[:, None]) % nt
    return np.take_along_axis(dyn, idx, axis=1).sum(axis=0)


def _best_boxcar_snr(ts, widths=(1, 2, 4, 8, 16, 32)):
    """Maximo S/N de cajas de anchura w (sumas acumuladas; ruido por MAD)."""
    med = np.median(ts)
    x = ts - med
    sd = 1.4826 * np.median(np.abs(x)) or ts.std() or 1.0
    cs = np.concatenate([[0.0], np.cumsum(x)])
    best = (-np.inf, 1, 0)
    for w in widths:
        if w > len(ts) // 4:
            break
        sm = (cs[w:] - cs[:-w]) / (sd * np.sqrt(w))
        i = int(np.argmax(sm))
        if sm[i] > best[0]:
            best = (float(sm[i]), w, i + w // 2)
    return best


def dispersion_search(dyn, freqs_ghz, dt, dm_max=1000.0, n_dm=200, n_null=19, seed=0):
    """dyn: canales x tiempo; freqs_ghz: frecuencia de cada canal."""
    dyn = np.asarray(dyn, dtype=float)
    dyn = (dyn - np.median(dyn, axis=1, keepdims=True))
    sd = 1.4826 * np.median(np.abs(dyn), axis=1, keepdims=True)
    dyn = dyn / np.where(sd > 0, sd, 1.0)
    f = np.asarray(freqs_ghz, dtype=float)
    f_ref = f.max()
    dms = np.linspace(0, dm_max, n_dm)

    def scan(d, curve=None):
        best = (-np.inf, 0.0, 1, 0)
        for dm in dms:
            delays = np.round(K_DM_S * dm * (f ** -2 - f_ref ** -2) / dt)
            snr, w, i = _best_boxcar_snr(_dedisperse(d, delays))
            if curve is not None:
                curve.append(snr)
            if snr > best[0]:
                best = (snr, float(dm), w, i)
        return best

    from scipy.stats import norm
    dm_curve = []
    snr, dm, w, i = scan(dyn, dm_curve)
    best_delays = np.round(K_DM_S * dm * (f ** -2 - f_ref ** -2) / dt)
    series = _dedisperse(dyn, best_delays)
    n_trials = len(dms) * dyn.shape[1] * 6                  # DM x tiempo x anchuras de caja
    fap = float(min(1.0, n_trials * norm.sf(snr)))          # Bonferroni (conservador: los ensayos se solapan)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(int(n_null)):
        sh = np.stack([np.roll(row, rng.integers(dyn.shape[1])) for row in dyn])
        null.append(scan(sh)[0])
    null = np.array(null)
    zero_dm = dm <= dms[1] * 1.5
    return {"snr": snr, "dm": dm, "width_samples": int(w), "t_peak_s": float(i * dt), "fap": fap,
            "n_trials": int(n_trials), "null_snr": null.tolist(), "dm_grid": dms.tolist(), "dm_curve": dm_curve,
            "dedispersed": series.tolist(),
            # FAP analitica y ademas superar todos los nulos permutados (colas pesadas del ruido real).
            "detected": bool(fap <= FAP and snr > null.max()),
            "likely_rfi": bool(zero_dm),
            "note": ("DM ≈ 0: llega a todas las frecuencias a la vez → probable interferencia terrestre." if zero_dm
                     else "DM = %.1f pc/cm³: retraso f⁻² compatible con plasma entre emisor y antena." % dm),
            "method": "desdispersión incoherente %d DM × cajas; FAP = N·P(Z>snr) con N = %d; control: %d nulos "
                      "permutados por canal" % (n_dm, n_trials, n_null)}


# ---------------------------------------------------------------- 3. deriva Doppler

def drift_search(dyn, df_hz, dt, max_drift=1.0, n_null=19, seed=0):
    """dyn: tiempo x canales. Desplaza cada espectro según la deriva y suma (de-Doppler).
    Rejilla: paso = df / (T_total), un canal en toda la observación (la resolución real;
    con pasos más gruesos una deriva intermedia se difumina: medido, -0.15 Hz/s no se
    detectaba con pasos de 0.1 Hz/s)."""
    dyn = np.asarray(dyn, dtype=float)
    dyn = dyn - np.median(dyn, axis=1, keepdims=True)
    sd = 1.4826 * np.median(np.abs(dyn), axis=1, keepdims=True)
    dyn = dyn / np.where(sd > 0, sd, 1.0)
    nt, nch = dyn.shape
    step = df_hz / (nt * dt)
    n_side = int(np.ceil(max_drift / step))
    drifts = np.arange(-n_side, n_side + 1) * step
    base = np.arange(nch)[None, :]
    offsets = [np.round(r * np.arange(nt) * dt / df_hz).astype(int) for r in drifts]

    def scan(d):
        best = (-np.inf, 0.0, 0)
        for r, o in zip(drifts, offsets):
            acc = np.take_along_axis(d, (base + o[:, None]) % nch, axis=1).sum(axis=0) / np.sqrt(nt)
            j = int(np.argmax(acc))
            if acc[j] > best[0]:
                best = (float(acc[j]), float(r), j)
        return best

    from scipy.stats import norm
    snr, drift, ch = scan(dyn)
    n_trials = len(drifts) * nch
    fap = float(min(1.0, n_trials * norm.sf(snr)))          # Bonferroni sobre deriva x canal
    rng = np.random.default_rng(seed)
    null = np.array([scan(np.stack([np.roll(row, rng.integers(nch)) for row in dyn]))[0] for _ in range(int(n_null))])
    zero = abs(drift) < step * 0.5
    return {"snr": snr, "drift_hz_s": drift, "drift_step_hz_s": float(step), "channel": ch, "fap": fap,
            "null_snr": null.tolist(), "n_trials": int(n_trials),
            # Dos condiciones: FAP analitica (ruido gaussiano) y superar todos los nulos
            # permutados (protege si el ruido real tiene colas pesadas).
            "detected": bool(fap <= FAP and snr > null.max()),
            "likely_rfi": bool(zero),
            "note": ("Deriva ≈ 0 Hz/s: emisor fijo respecto a la antena → probable interferencia terrestre." if zero
                     else "Deriva %.4f Hz/s: hay aceleración relativa emisor-antena (Doppler)." % drift),
            "method": "de-Doppler %d derivas (paso %.4f Hz/s); FAP = N·P(Z>snr) con N = %d; control: %d nulos permutados"
                      % (len(drifts), step, n_trials, n_null)}


# ---------------------------------------------------------------- cruce con catálogo público

def crossmatch_catalog(rows, period_s=None, dm=None, p_tol=0.002, dm_tol=0.1, period_err_s=None):
    """rows: dicts de un CSV exportado (p. ej. ATNF). Busca P0 (s) y DM con tolerancias
    relativas; acepta armónicos P/2, 2P, P/3, 3P. Si se da `period_err_s`, la tolerancia del
    periodo es max(p_tol, 3 * error relativo): el periodo medido no es más preciso que eso."""
    if period_s and period_err_s:
        p_tol = max(p_tol, 3.0 * period_err_s / period_s)
    def num(r, *names):
        for nm in names:
            for k, v in r.items():
                if str(k).strip().lower() == nm:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return None
        return None

    matches = []
    for r in rows:
        p0, d0 = num(r, "p0", "period", "p0_s"), num(r, "dm")
        ok_p = ok_dm = None
        harm = None
        if period_s and p0:
            for h in (1, 2, 0.5, 3, 1 / 3):
                if abs(period_s * h / p0 - 1) <= p_tol:
                    ok_p, harm = True, h
                    break
            ok_p = bool(ok_p)
        if dm is not None and d0 is not None:
            ok_dm = abs(dm - d0) <= dm_tol * max(d0, 1.0)
        if (ok_p is None or ok_p) and (ok_dm is None or ok_dm) and (ok_p or ok_dm):
            matches.append({"row": r, "harmonic": harm})
    return matches
