"""OLED drift-diffusion DC solver. Units: cm, s, V, cm^-3, A/cm^2.

The original upwind/hopping model is retained for nonnegative densities.
Defaults retain its Poisson voltage quadrature; see README.md for options.
S and T do not feed back into p and n in this model: DC carrier solves need
2N states, not 4N. full_rhs supports exciton transients.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import time
import warnings
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits


@dataclass
class Config:
    grid_nm: float = 1.0
    temperature: float = 300.0
    eps_r: float = 3.5
    vbi: float = 0.0
    density_scale: float = 1e17
    rtol: float = 1e-5
    atol: float = 1e-10  # scaled density, equivalent to 1e7 cm^-3
    max_time: float = 0.5
    steady_rate: float = 1e-2  # max |dy/dt|/(|y|+steady_floor/scale), s^-1
    steady_floor_cm3: float = 1e10  # Avoid declaring failure over negligible CTL densities.
    current_rtol: float = 1e-3
    poisson: str = 'legacy'  # 'trapezoid' changes discrete voltage constraint
    field_lowering: bool = True
    triplet_loss: str = 'corrected'  # corrected -kT*T; legacy is -kT*S


class Device:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        c = self.cfg
        if c.grid_nm <= 0 or c.temperature <= 0 or c.eps_r <= 0:
            raise ValueError('grid_nm, temperature, eps_r must be positive')
        if c.density_scale <= 0 or c.atol <= 0 or c.rtol <= 0 or c.max_time <= 0:
            raise ValueError('Solver scales, tolerances and max_time must be positive')
        if c.poisson not in ('legacy', 'trapezoid') or c.triplet_loss not in ('legacy', 'corrected'):
            raise ValueError('Invalid model option')
        self.e = 1.6022e-19
        self.eps = c.eps_r * 8.85e-14
        self.vt = 1.3807e-23 * c.temperature / self.e
        self.dx = c.grid_nm * 1e-7
        self.thickness = np.array([60., 5., 30., 5., 31.])
        counts = np.rint(self.thickness / c.grid_nm).astype(int)
        if np.any(counts < 1) or not np.allclose(counts*c.grid_nm, self.thickness, atol=1e-10, rtol=0):
            raise ValueError('Every layer thickness must be an integer multiple of grid_nm')
        self.N = int(counts.sum())
        self.x_nm = (np.arange(self.N)+0.5)*c.grid_nm
        self.eml = (self.x_nm > 65) & (self.x_nm < 95)
        self.edges = np.cumsum(counts)[:-1]  # interface field index; left cell index = edge-1
        self.mup = np.repeat([5e-5, 1e-5, 1e-6, 1e-8, 1e-8], counts)[:, None]
        self.mun = np.repeat([1e-8, 1e-8, 1e-5, 5e-6, 1e-5], counts)[:, None]
        self.gp = np.full((self.N, 1), 1e-3)
        self.gn = self.gp.copy()
        self.hb = -np.diff([-5.15, -5.52, -5.52, -6.50, -6.50])
        self.lb = -np.diff([-1.80, -1.91, -2.75, -2.72, -2.72])
        self.rc = self.e/(4*np.pi*self.eps*self.vt)
        self.contact_prefactor = 16*np.pi*self.eps*self.vt**2

    def field(self, p, n, voltage):
        q = np.vstack((np.zeros((1, p.shape[1])), np.cumsum(p-n, axis=0)))
        q *= self.e*self.cfg.density_scale*self.dx/self.eps
        if self.cfg.poisson == 'legacy':
            offset = (voltage-self.cfg.vbi-self.dx*q.sum(axis=0))/(self.N*self.dx)
        else:
            offset = (voltage-self.cfg.vbi-self.dx*(q.sum(axis=0)-0.5*(q[0]+q[-1])))/(self.N*self.dx)
        fi = q+offset
        return 0.5*(fi[:-1]+fi[1:]), fi

    def contact(self, mu, density, field, barrier, favored, orientation):
        # density is scaled; stable SE/S0=(1+a)*(1+a+2*s)/4, a=sqrt(1+2*s).
        s = np.sqrt(np.abs(field)*self.rc/self.vt)
        a = np.sqrt(1+2*s)
        se = (1+a)*(1+a+2*s)/4
        exponent = -barrier/self.vt + np.where(favored, s, -s*s/4)
        with np.errstate(over='raise', invalid='raise'):
            source = (1e21/self.cfg.density_scale)*np.exp(exponent)
            sink = np.where(favored, se, 1+s*s/4)
            return orientation*self.contact_prefactor*mu*(source-density*sink)/(self.e*self.dx)

    def transport(self, y, voltage):
        # y is (2N,k), enabling batched numerical Jacobian columns.
        p, n = y[:self.N], y[self.N:2*self.N]
        f, fi = self.field(p, n, voltage)
        with np.errstate(over='raise', invalid='raise'):
            up = self.mup*np.exp(self.gp*np.sqrt(np.abs(f)))
            un = self.mun*np.exp(self.gn*np.sqrt(np.abs(f)))
        # Positive forward/reverse rates times signed density: smooth through p=n=0.
        # For p,n>=0 this is algebraically identical to supplied Drift+Diff.
        pd = up*self.vt/self.dx**2*p
        nd = un*self.vt/self.dx**2*n
        pf = pd+up*np.maximum(f, 0)/self.dx*p
        pr = pd+up*np.maximum(-f, 0)/self.dx*p
        nf = nd+un*np.maximum(-f, 0)/self.dx*n
        nr = nd+un*np.maximum(f, 0)/self.dx*n
        for i, edge in enumerate(self.edges):
            lowering = fi[edge]*self.dx if self.cfg.field_lowering else 0.0
            he, le = self.hb[i]-lowering, self.lb[i]-lowering
            if self.hb[i] != 0:
                pf[edge-1] *= np.exp(-np.maximum(he, 0)/self.vt)
                pr[edge] *= np.exp(np.minimum(he, 0)/self.vt)
            if self.lb[i] != 0:
                nr[edge] *= np.exp(-np.maximum(le, 0)/self.vt)
                nf[edge-1] *= np.exp(np.minimum(le, 0)/self.vt)
        # Hole/electron PARTICLE flux, positive towards cathode; scaled cm^-3/s.
        qp0 = self.contact(up[0], p[0], fi[0], .3, fi[0] > 0, +1)
        qpL = self.contact(up[-1], p[-1], fi[-1], 1., fi[-1] < 0, -1)
        qn0 = self.contact(un[0], n[0], fi[0], 1., fi[0] < 0, +1)
        qnL = self.contact(un[-1], n[-1], fi[-1], .3, fi[-1] > 0, -1)
        qp = np.vstack((qp0, pf[:-1]-pr[1:], qpL))
        qn = np.vstack((qn0, nf[:-1]-nr[1:], qnL))
        r = self.e*(up+un)/self.eps*self.cfg.density_scale*p*n
        return qp, qn, r, f, fi

    def rhs(self, t, y, voltage):
        one = y.ndim == 1
        z = y[:, None] if one else y
        qp, qn, r, _, _ = self.transport(z, voltage)
        out = np.vstack((qp[:-1]-qp[1:]-r, qn[:-1]-qn[1:]-r))
        return out[:, 0] if one else out

    def full_rhs(self, t, y, voltage):
        one = y.ndim == 1
        z = y[:, None] if one else y
        pn = z[:2*self.N]
        s, tr = z[2*self.N:3*self.N], z[3*self.N:]
        qp, qn, r, _, _ = self.transport(pn, voltage)
        carriers = pn[:self.N]+pn[self.N:]
        scale = self.cfg.density_scale
        ds = .25*r-(1.08e8+3.71e7+1.57e8)*s+4.33e5*tr-1e-12*scale*s*tr-1e-12*scale*s*carriers
        loss = tr if self.cfg.triplet_loss == 'corrected' else s
        dt = .75*r-8.28e4*loss+1.57e8*s-4.33e5*tr-1e-13*scale*tr**2-5e-14*scale*tr*carriers
        out = np.vstack((qp[:-1]-qp[1:]-r, qn[:-1]-qn[1:]-r, ds, dt))
        return out[:, 0] if one else out

    def diagnostics(self, y, voltage):
        qp, qn, r, f, fi = self.transport(y[:, None], voltage)
        jp = qp[:, 0]*self.e*self.dx*self.cfg.density_scale
        jn = -qn[:, 0]*self.e*self.dx*self.cfg.density_scale
        j = jp+jn
        rate = np.max(np.abs(self.rhs(0, y, voltage))/(np.abs(y)+self.cfg.steady_floor_cm3/self.cfg.density_scale))
        spread = np.ptp(j)/max(np.max(np.abs(j)), 1e-9)
        return dict(voltage=float(voltage), current_mA_cm2=float(np.median(.5*(j[:-1]+j[1:]))*1e3),
                    steady_rate=float(rate), current_spread=float(spread), min_density=float(y.min()*self.cfg.density_scale),
                    field_min=float(fi.min()), field_max=float(fi.max()))

    def solve_voltage(self, voltage, initial=None, method='BDF', early_stop=True, vectorized=True):
        y = np.full(2*self.N, 1e-5/self.cfg.density_scale) if initial is None else np.array(initial, float).copy()
        if y.shape != (2*self.N,) or not np.all(np.isfinite(y)) or np.any(y < 0):
            raise ValueError('initial must be a finite nonnegative 2N-vector of SCALED densities')
        c = self.cfg
        schedule = np.unique(np.r_[np.array([1e-6, 1e-5, 1e-4, 1e-3, .01, .1])[np.array([1e-6, 1e-5, 1e-4, 1e-3, .01, .1]) < c.max_time], c.max_time]) if early_stop else [c.max_time]
        start = time.perf_counter()
        t0 = 0.0
        nfev = njev = nlu = 0
        for tend in schedule:
            with threadpool_limits(limits=1, user_api='blas'):
                sol = solve_ivp(lambda t, z: self.rhs(t, z, voltage), (t0, tend), y,
                                method=method, rtol=c.rtol, atol=c.atol, vectorized=vectorized,
                                t_eval=[tend])
            if not sol.success or len(sol.t) == 0 or sol.t[-1] < tend*(1-1e-12):
                raise RuntimeError(f'{voltage:g} V: integration failed: {sol.message}')
            y = sol.y[:, -1]
            if not np.all(np.isfinite(y)) or y.min() < -100*c.atol:
                raise RuntimeError(f'{voltage:g} V: nonfinite or materially negative density')
            # Only remove accepted endpoint roundoff; never clip internal solver states.
            y = np.maximum(y, 0)
            nfev += sol.nfev; njev += sol.njev; nlu += sol.nlu
            d = self.diagnostics(y, voltage)
            t0 = float(tend)
            if early_stop and d['steady_rate'] <= c.steady_rate and d['current_spread'] <= c.current_rtol:
                break
        d.update(elapsed_seconds=time.perf_counter()-start, integration_time=t0,
                 nfev=nfev, njev=njev, nlu=nlu,
                 steady=bool(d['steady_rate'] <= c.steady_rate and d['current_spread'] <= c.current_rtol))
        if not d['steady']:
            warnings.warn(f'{voltage:g} V reached time limit without DC convergence: {d}', RuntimeWarning)
        return y, d

    def exciton_steady(self, y, voltage):
        """Independent local steady excitons, NOT finite-time transient S,T."""
        r = self.transport(y[:, None], voltage)[2][:, 0]*self.cfg.density_scale
        total = (y[:self.N]+y[self.N:])*self.cfg.density_scale
        ss = np.zeros(self.N); tt = ss.copy()
        for i, (ri, ci) in enumerate(zip(r, total)):
            if ri <= 0:
                continue
            a = 1.08e8+3.71e7+1.57e8+1e-12*ci
            local_scale = ri/4.33e5  # Per-cell scaling also resolves negligible CTL populations.
            def singlet(t):
                return (.25*ri+4.33e5*t)/(a+1e-12*t)
            def balance_scaled(z):
                t = z*local_scale
                s = singlet(t)
                loss = t if self.cfg.triplet_loss == 'corrected' else s
                return (.75*ri-8.28e4*loss+1.57e8*s-4.33e5*t-1e-13*t*t-5e-14*t*ci)/ri
            upper = 1.0
            for _ in range(100):
                if balance_scaled(upper) <= 0:
                    break
                upper *= 2
            else:
                raise RuntimeError('Cannot bracket exciton steady state')
            tt[i] = brentq(balance_scaled, 0., upper, xtol=1e-14, rtol=1e-12)*local_scale
            ss[i] = singlet(tt[i])
        return ss, tt


def sweep(device, voltages, continuation=True, **kwargs):
    states, records = [], []
    y = None
    for v in voltages:
        y, d = device.solve_voltage(float(v), y if continuation else None, **kwargs)
        states.append(y.copy()); records.append(d)
        print(f"V={v: .4f}, J={d['current_mA_cm2']: .6g} mA/cm2, steady={d['steady']}, {d['elapsed_seconds']:.3f}s", flush=True)
    return np.array(states), records


def solve_target(device, voltages, states, records, target, current_tol=1e-4):
    """Bracket a target current and RE-SOLVE carriers at every trial voltage."""
    for i, d in enumerate(records):
        if d['steady'] and abs(d['current_mA_cm2']-target) <= current_tol:
            return states[i].copy(), d.copy()
    for i in range(len(voltages)-1):
        if not (records[i]['steady'] and records[i+1]['steady']):
            continue
        ja, jb = records[i]['current_mA_cm2'], records[i+1]['current_mA_cm2']
        if (ja-target)*(jb-target) > 0:
            continue
        va, vb = float(voltages[i]), float(voltages[i+1])
        ya, yb = states[i].copy(), states[i+1].copy()
        for _ in range(40):
            vm = .5*(va+vb)
            ym, dm = device.solve_voltage(vm, .5*(ya+yb))
            if not dm['steady']:
                raise RuntimeError('Target solve did not reach DC convergence')
            jm = dm['current_mA_cm2']
            if abs(jm-target) <= current_tol:
                return ym, dm
            if (ja-target)*(jm-target) <= 0:
                vb, yb, jb = vm, ym, jm
            else:
                va, ya, ja = vm, ym, jm
        raise RuntimeError('Target current did not converge within 40 bisections')
    raise ValueError(f'Target {target:g} mA/cm2 is not bracketed by converged sweep points; extend the voltage range')


def save_profile(device, y, d, folder, name='profile'):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    qp, qn, r, f, fi = device.transport(y[:, None], d['voltage'])
    s, t = device.exciton_steady(y, d['voltage'])
    r = r[:, 0]*device.cfg.density_scale
    area = np.trapezoid(r[device.eml], device.x_nm[device.eml])
    normalized = np.full(device.N, np.nan)
    if area > 0:
        normalized[device.eml] = r[device.eml]/area
    data = np.column_stack((device.x_nm, y[:device.N]*device.cfg.density_scale, y[device.N:]*device.cfg.density_scale,
                            r, f[:, 0], s, t, normalized))
    np.savetxt(folder/(name+'.csv'), data, delimiter=',', comments='',
               header='x_nm,p_cm-3,n_cm-3,R_cm-3_s-1,F_V_cm-1,S_DC_cm-3,T_DC_cm-3,R_EML_normalized_nm-1')
    np.savez(folder/(name+'.npz'), state_scaled=y, profile=data, field_interface=fi[:, 0],
             Jp_interface=qp[:, 0]*device.e*device.dx*device.cfg.density_scale,
             Jn_interface=-qn[:, 0]*device.e*device.dx*device.cfg.density_scale)
    return data



def plot_results(device, voltages, records, profile, profile_info, output_folder):
    """Create a six-panel OLED device-physics summary and individual figures."""
    import matplotlib.pyplot as plt

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    x = device.x_nm
    eml = device.eml
    j = np.array([d['current_mA_cm2'] for d in records])

    p_density = profile[:, 1]
    n_density = profile[:, 2]
    recombination = profile[:, 3]
    field = profile[:, 4]
    singlet = profile[:, 5]
    triplet = profile[:, 6]
    recombination_norm = profile[:, 7]

    voltage = profile_info['voltage']
    current = profile_info['current_mA_cm2']

    # Layer boundaries for visual guidance.
    boundaries = np.cumsum(device.thickness)[:-1]

    # --- Six-panel dashboard ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    # 1) J-V
    ax = axes[0, 0]
    ax.plot(voltages, j, 'o-')
    ax.axvline(voltage, linestyle='--', linewidth=1)
    ax.axhline(current, linestyle='--', linewidth=1)
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current density (mA cm$^{-2}$)')
    ax.set_title('J-V characteristics')
    ax.grid(alpha=0.25)

    # 2) Carrier densities
    ax = axes[0, 1]
    ax.semilogy(x, np.maximum(p_density, 1.0), label='Holes, p')
    ax.semilogy(x, np.maximum(n_density, 1.0), label='Electrons, n')
    for b in boundaries:
        ax.axvline(b, linewidth=0.8, alpha=0.35)
    ax.set_xlabel('Position (nm)')
    ax.set_ylabel('Carrier density (cm$^{-3}$)')
    ax.set_title('Carrier distributions')
    ax.legend()
    ax.grid(alpha=0.25)

    # 3) Electric field
    ax = axes[0, 2]
    ax.plot(x, field)
    for b in boundaries:
        ax.axvline(b, linewidth=0.8, alpha=0.35)
    ax.set_xlabel('Position (nm)')
    ax.set_ylabel('Electric field (V cm$^{-1}$)')
    ax.set_title('Electric-field distribution')
    ax.grid(alpha=0.25)

    # EML-relative coordinate, useful for RZ and excitons.
    x_eml = x[eml] - x[eml][0]

    # 4) Recombination zone
    ax = axes[1, 0]
    ax.plot(x_eml, recombination_norm[eml])
    ax.set_xlabel('Position in EML (nm)')
    ax.set_ylabel('Normalized recombination (nm$^{-1}$)')
    ax.set_title('Recombination-zone profile')
    ax.grid(alpha=0.25)

    # 5) Singlet exciton density
    ax = axes[1, 1]
    ax.semilogy(x_eml, np.maximum(singlet[eml], 1.0))
    ax.set_xlabel('Position in EML (nm)')
    ax.set_ylabel('Singlet density (cm$^{-3}$)')
    ax.set_title('Singlet exciton distribution')
    ax.grid(alpha=0.25)

    # 6) Triplet exciton density
    ax = axes[1, 2]
    ax.semilogy(x_eml, np.maximum(triplet[eml], 1.0))
    ax.set_xlabel('Position in EML (nm)')
    ax.set_ylabel('Triplet density (cm$^{-3}$)')
    ax.set_title('Triplet exciton distribution')
    ax.grid(alpha=0.25)

    fig.suptitle(
        f'OLED drift-diffusion simulation: '
        f'{voltage:.4f} V, {current:.4f} mA cm$^{{-2}}$',
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_folder/'summary_dashboard.png', dpi=200, bbox_inches='tight')

    # --- Individual figures: useful for papers, presentations, and GitHub README ---
    def save_single(filename, title, xlabel, ylabel, xdata, ydata, *, semilogy=False, labels=None):
        f, ax = plt.subplots(figsize=(6.2, 4.5))
        if ydata.ndim == 1:
            if semilogy:
                ax.semilogy(xdata, np.maximum(ydata, 1.0))
            else:
                ax.plot(xdata, ydata)
        else:
            for k in range(ydata.shape[1]):
                yy = np.maximum(ydata[:, k], 1.0) if semilogy else ydata[:, k]
                if semilogy:
                    ax.semilogy(xdata, yy, label=labels[k] if labels else None)
                else:
                    ax.plot(xdata, yy, label=labels[k] if labels else None)
            if labels:
                ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        f.tight_layout()
        f.savefig(output_folder/filename, dpi=200, bbox_inches='tight')

    save_single('JV.png', 'J-V characteristics', 'Voltage (V)',
                'Current density (mA cm$^{-2}$)', voltages, j)
    save_single('carrier_density.png', 'Carrier distributions', 'Position (nm)',
                'Carrier density (cm$^{-3}$)', x,
                np.column_stack((p_density, n_density)), semilogy=True,
                labels=['Holes, p', 'Electrons, n'])
    save_single('electric_field.png', 'Electric-field distribution', 'Position (nm)',
                'Electric field (V cm$^{-1}$)', x, field)
    save_single('recombination_zone.png', 'Recombination-zone profile', 'Position in EML (nm)',
                'Normalized recombination (nm$^{-1}$)', x_eml, recombination_norm[eml])
    save_single('singlet_distribution.png', 'Singlet exciton distribution', 'Position in EML (nm)',
                'Singlet density (cm$^{-3}$)', x_eml, singlet[eml], semilogy=True)
    save_single('triplet_distribution.png', 'Triplet exciton distribution', 'Position in EML (nm)',
                'Triplet density (cm$^{-3}$)', x_eml, triplet[eml], semilogy=True)

    # In Spyder this sends the figures to the Plots pane (or opens figure windows,
    # depending on the selected graphics backend).
    plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vmin', type=float, default=-.4)
    p.add_argument('--vmax', type=float, default=2.)
    p.add_argument('--vstep', type=float, default=.1)
    p.add_argument('--target', type=float, default=3.)
    p.add_argument('--output', default='results')
    p.add_argument('--poisson', choices=['legacy', 'trapezoid'], default='legacy')
    p.add_argument('--triplet-loss', choices=['legacy', 'corrected'], default='corrected')
    p.add_argument('--max-time', type=float, default=.5)
    p.add_argument('--no-plot', action='store_true',
                   help='Do not create/show plots (plots are on by default for Spyder-friendly use).')
    args = p.parse_args()

    if args.vstep <= 0 or args.vmax < args.vmin:
        p.error('Require vstep>0 and vmax>=vmin')

    dev = Device(Config(poisson=args.poisson,
                        triplet_loss=args.triplet_loss,
                        max_time=args.max_time))
    vs = np.arange(args.vmin, args.vmax+args.vstep*1e-6, args.vstep)
    states, ds = sweep(dev, vs)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'sweep.npz', voltages=vs, states_scaled=states)
    np.savetxt(out/'JV.csv',
               np.column_stack((vs, [d['current_mA_cm2'] for d in ds])),
               delimiter=',', header='V,J_mA_cm2', comments='')

    target_info = None
    target_profile = None
    target_state = None

    try:
        target_state, target_info = solve_target(dev, vs, states, ds, args.target)
        target_profile = save_profile(dev, target_state, target_info, out, 'target_profile')
        print(f"Target: {target_info['voltage']:.7g} V, "
              f"{target_info['current_mA_cm2']:.7g} mA/cm2")
    except ValueError as exc:
        warnings.warn(str(exc))

    last_profile = save_profile(dev, states[-1], ds[-1], out, 'last_voltage_profile')

    (out/'diagnostics.json').write_text(
        json.dumps(dict(config=vars(dev.cfg), sweep=ds, target=target_info), indent=2)
    )

    # Prefer the requested current-density state for spatial distributions.
    # Fall back to the final voltage if the requested target is outside the sweep.
    if target_profile is not None:
        plot_profile = target_profile
        plot_info = target_info
        plot_source = f'target current {args.target:g} mA/cm2'
    else:
        plot_profile = last_profile
        plot_info = ds[-1]
        plot_source = f'last sweep voltage {vs[-1]:g} V'

    print(f'Profile plots use: {plot_source}')
    print(f'Results saved to: {out.resolve()}')

    if not args.no_plot:
        plot_results(dev, vs, ds, plot_profile, plot_info, out)


if __name__ == '__main__':
    main()