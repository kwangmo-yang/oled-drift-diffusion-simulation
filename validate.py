"""Run numerical equivalence, convergence, regression and timing checks.

This is a Python validation; it does not claim a MATLAB/ode15s execution.
Run from the repository root: python validate.py
"""
from pathlib import Path
import json
import time
import platform
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.io import savemat
from threadpoolctl import threadpool_limits
from recombination import Config, Device, sweep, solve_target


def legacy_rhs_physical(y, voltage, d):
    """Independent literal translation of uploaded 4N RHS on positive states.
    Original contact psi is deliberately retained for equivalence testing.
    """
    N = d.N; e = d.e; dx = d.dx; vt = d.vt; eps = d.eps
    p, n, S, T = np.split(y, 4)
    fi = np.cumsum(np.r_[0., e*(p-n)])*dx/eps
    fi += (voltage-d.cfg.vbi-np.sum(fi)*dx)/(N*dx)
    f = (fi[:-1]+fi[1:])/2
    up = d.mup[:, 0]*np.exp(d.gp[:, 0]*np.sqrt(np.abs(f)))
    un = d.mun[:, 0]*np.exp(d.gn[:, 0]*np.sqrt(np.abs(f)))
    pd = up*f*p/dx; nd = un*f*n/dx
    pf = (np.sign(pd)*pd+pd)/2+np.abs(up*vt*p/dx**2)
    pr = (np.sign(pd)*pd-pd)/2+np.abs(up*vt*p/dx**2)
    nf = (np.sign(nd)*nd-nd)/2+np.abs(un*vt*n/dx**2)
    nr = (np.sign(nd)*nd+nd)/2+np.abs(un*vt*n/dx**2)
    def contact(mu, den, field, barrier, favored, sign):
        s0 = 16*np.pi*eps*vt**2*mu
        s1 = (16*np.pi*eps*vt**2+e*abs(field))*mu
        z = abs(field*d.rc/vt)
        psi = (1-np.sqrt(1+2*np.sqrt(z)))/z+1/np.sqrt(z)
        se = s0*(1/psi**2-z)/4
        if favored:
            return sign*(s0*1e21*np.exp(-barrier/vt)*np.exp(np.sqrt(z))-den*se)
        return sign*(s0*1e21*np.exp(-barrier/vt-z/4)-den*s1)
    jp0 = contact(up[0],p[0],fi[0],.3,fi[0]>0,1)
    jpL = contact(up[-1],p[-1],fi[-1],1,fi[-1]<0,-1)
    jnL = contact(un[-1],n[-1],fi[-1],.3,fi[-1]>0,1)
    jn0 = contact(un[0],n[0],fi[0],1,fi[0]<0,-1)
    for i, edge in enumerate(d.edges):
        lowering = fi[edge]*dx if d.cfg.field_lowering else 0
        he, le = d.hb[i]-lowering, d.lb[i]-lowering
        if d.hb[i] != 0:
            if he > 0: pf[edge-1] *= np.exp(-he/vt)
            elif he < 0: pr[edge] *= np.exp(he/vt)
        if d.lb[i] != 0:
            if le > 0: nr[edge] *= np.exp(-le/vt)
            elif le < 0: nf[edge-1] *= np.exp(le/vt)
    dp = np.empty(N); dn = np.empty(N)
    dp[-1] = pf[-2]-pr[-1]-jpL/e/dx
    dp[0] = -pf[0]+pr[1]+jp0/e/dx
    dp[1:-1] = -pf[1:-1]-pr[1:-1]+pf[:-2]+pr[2:]
    dn[-1] = nf[-2]-nr[-1]+jnL/e/dx
    dn[0] = -nf[0]+nr[1]-jn0/e/dx
    dn[1:-1] = -nf[1:-1]-nr[1:-1]+nf[:-2]+nr[2:]
    R = e*(up+un)/eps*p*n
    ds = R/4-(1.08e8+3.71e7)*S-1.57e8*S+4.33e5*T-1e-12*S*T-1e-12*S*(n+p)
    dt = 3*R/4-8.28e4*S+1.57e8*S-4.33e5*T-1e-13*T*T-5e-14*T*(n+p)
    return np.r_[dp-R,dn-R,ds,dt]


def relative(a, b, floor=1e-9):
    return float(np.max(np.abs(a-b)/(np.abs(b)+floor)))


def main():
    out = Path(__file__).resolve().parent/'validation'
    out.mkdir(exist_ok=True)
    report = {'environment': dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                  platform=platform.platform(), blas_threads=1), 'checks': {}}
    c = Config(triplet_loss='legacy')
    d = Device(c); rng = np.random.default_rng(4931)
    fixture_y = 10**rng.uniform(11,16,size=4*d.N)/c.density_scale
    old = legacy_rhs_physical(fixture_y*c.density_scale,1.4,d)/c.density_scale
    new = d.full_rhs(0,fixture_y,1.4)
    err = np.max(np.abs(old-new))/np.max(np.abs(old))
    assert err < 1e-11, err
    np.testing.assert_allclose(new,old,rtol=3e-10,atol=1e-7)
    report['checks']['legacy_positive_state_rhs_relative_inf_error'] = float(err)
    batch = np.column_stack([fixture_y[:2*d.N],fixture_y[:2*d.N]*1.01])
    np.testing.assert_allclose(d.rhs(0,batch,1.4), np.column_stack([d.rhs(0,z,1.4) for z in batch.T]),rtol=1e-13)
    assert np.all(np.isfinite(d.rhs(0,np.zeros(2*d.N),0)))
    # Zero and near-zero positive and negative contact fields; no singular SE.
    fields = np.r_[-np.logspace(-20,8,30),0.,np.logspace(-20,8,30)]
    for orientation in [-1,1]:
        assert np.all(np.isfinite(d.contact(np.ones(fields.size)*1e-5,np.ones(fields.size),fields,.3,fields>0,orientation)))
    report['checks']['zero_field_and_vectorization'] = 'passed'
    # Conservation: internal particle flux telescopes exactly up to roundoff.
    z = fixture_y[:2*d.N]
    qp,qn,R,_,_ = d.transport(z[:,None],1.4)
    rhs = d.rhs(0,z,1.4)
    np.testing.assert_allclose(rhs[:d.N].sum(),(qp[0]-qp[-1]-R.sum())[0],rtol=1e-12,atol=1e-7)
    np.testing.assert_allclose(rhs[d.N:].sum(),(qn[0]-qn[-1]-R.sum())[0],rtol=1e-12,atol=1e-7)
    report['checks']['discrete_particle_conservation'] = 'passed'
    # Physical voltage integral: legacy discrepancy exposed, optional fix verified.
    legacy_f, _ = d.field(z[:d.N,None],z[d.N:,None],1.4)
    dc = Device(Config(poisson='trapezoid'))
    corrected_f, _ = dc.field(z[:d.N,None],z[d.N:,None],1.4)
    assert abs(corrected_f.sum()*dc.dx-1.4) < 1e-12
    report['checks']['legacy_voltage_integral_error_V'] = float(legacy_f.sum()*d.dx-1.4)
    report['checks']['trapezoid_voltage_integral_error_V'] = float(corrected_f.sum()*dc.dx-1.4)
    # Check non-1nm grid uses the same spacing in all terms.
    coarse = Device(Config(grid_nm=0.5))
    assert coarse.N==262 and np.all(np.isfinite(coarse.rhs(0,np.zeros(2*coarse.N),0)))
    try: Device(Config(grid_nm=2))
    except ValueError: pass
    else: raise AssertionError('Invalid layer grid accepted')
    report['checks']['grid_consistency'] = 'passed'
    print('Algebra, conservation, zero-field and grid checks passed', flush=True)
    # Full requested grid, including a literal zero cold-start regression.
    dev = Device()
    vs = np.round(np.arange(-.4,2.0001,.1),10)
    states, records = sweep(dev,vs)
    assert all(r['steady'] for r in records)
    cold0, cold0d = dev.solve_voltage(0.)
    assert cold0d['steady']
    report['checks']['zero_voltage_cold_start'] = cold0d
    report['sweep'] = records
    np.savez(out/'validated_sweep.npz',voltages=vs,states_scaled=states)
    yt, dt = solve_target(dev,vs,states,records,3.)
    assert dt['steady'] and abs(dt['current_mA_cm2']-3)<1e-4
    report['target'] = dt
    for invalid_target in [-100,100]:
        try: solve_target(dev,vs,states,records,invalid_target)
        except ValueError: pass
        else: raise AssertionError('Unbracketed target was silently accepted')
    report['checks']['target_bracketing'] = 'passed'
    # Exciton DC residual in each equation relative to generation scale.
    S,T = dev.exciton_steady(yt,dt['voltage'])
    full = np.r_[yt,S/dev.cfg.density_scale,T/dev.cfg.density_scale]
    residual = dev.full_rhs(0,full,dt['voltage'])[2*dev.N:]
    gen = dev.transport(yt[:,None],dt['voltage'])[2][:,0]
    exerr = float(np.max(np.abs(residual)/(np.tile(gen,2)+1e-20)))
    assert exerr < 1e-7
    report['checks']['exciton_dc_residual_relative_to_generation'] = exerr
    # Tolerance tightening and early-stop check at physically relevant voltages.
    sensitivity=[]
    for v in [.4,1.2,2.]:
        idx=int(np.argmin(abs(vs-v))); base=states[idx]
        tightdev=Device(Config(rtol=1e-7,atol=1e-12,steady_rate=1e-3))
        tight, td=tightdev.solve_voltage(v,early_stop=False)
        assert td['steady']
        jerr=abs(td['current_mA_cm2']-records[idx]['current_mA_cm2'])/max(abs(td['current_mA_cm2']),1e-9)
        rb=dev.transport(base[:,None],v)[2][:,0]
        rt=tightdev.transport(tight[:,None],v)[2][:,0]
        rerr=float(np.max(abs(rb-rt))/max(np.max(abs(rt)),1e-20))
        assert jerr < 2e-3 and rerr < 2e-3, (jerr,rerr)
        sensitivity.append(dict(voltage=v,current_relative_difference=jerr,R_relative_inf_difference=rerr,
                                density_max_relative_difference_with_1e8_floor=relative(base,tight)))
    report['tolerance_sensitivity']=sensitivity
    print('25-point sweep, target and tolerance checks passed',flush=True)
    # Full 4N vs reduced 2N finite-time trajectory, same initial carriers.
    transient_times=np.array([1e-7,1e-6,1e-5,1e-4,1e-3])
    z2=np.full(2*dev.N,1e-5/dev.cfg.density_scale)
    z4=np.r_[z2,np.zeros(2*dev.N)]
    with threadpool_limits(limits=1,user_api='blas'):
        fullsol=solve_ivp(lambda t,z: dev.full_rhs(t,z,1.4),(0,transient_times[-1]),z4,method='BDF',
                          vectorized=True,rtol=1e-7,atol=1e-12,t_eval=transient_times)
        redsol=solve_ivp(lambda t,z: dev.rhs(t,z,1.4),(0,transient_times[-1]),z2,method='BDF',
                         vectorized=True,rtol=1e-7,atol=1e-12,t_eval=transient_times)
    assert fullsol.success and redsol.success
    trajectory_error=float(np.max(abs(fullsol.y[:2*dev.N]-redsol.y))/(np.max(abs(redsol.y))))
    assert trajectory_error < 2e-4, trajectory_error
    report['checks']['full_vs_reduced_transient_relative_inf_error'] = trajectory_error
    # Python/MATLAB cross-language fixtures, including literal original RHS.
    savemat(out/'reference_fixture.mat',dict(y_full_scaled=fixture_y[:,None],voltage=1.4,
             rhs_legacy_scaled=old[:,None],rhs_optimized_legacy_scaled=new[:,None],
             sweep_voltage=vs[None,:],sweep_state_scaled=states.T,
             sweep_current_mA_cm2=np.array([r['current_mA_cm2'] for r in records])[None,:]))
    # Benchmark numerical structure, NOT original MATLAB (not executable here).
    bvs=[.8,1.4,2.]
    modes=['full_4N_cold_fixed_time','reduced_2N_cold_fixed_time','reduced_2N_warm_fixed_time','optimized_2N_warm_early_stop']
    timings={mode:[] for mode in modes}; currents={}; control_diagnostics={}
    for rep in range(3):
        for mode in modes:
            tstart=time.perf_counter(); previous=None; js=[]; cds=[]
            for v in bvs:
                if mode.startswith('full'):
                    with threadpool_limits(limits=1,user_api='blas'):
                        sol=solve_ivp(lambda t,z:dev.full_rhs(t,z,v),(0,.5),z4,method='BDF',vectorized=True,
                                      rtol=dev.cfg.rtol,atol=dev.cfg.atol,t_eval=[.5])
                    assert sol.success
                    diag=dev.diagnostics(sol.y[:2*dev.N,-1],v)
                    diag['steady']=bool(diag['steady_rate']<=dev.cfg.steady_rate and diag['current_spread']<=dev.cfg.current_rtol)
                    current=diag['current_mA_cm2']
                else:
                    previous, diag=dev.solve_voltage(v,previous if 'warm' in mode else None,
                                                      early_stop=('early_stop' in mode))
                    if 'early_stop' in mode:
                        assert diag['steady']
                    current=diag['current_mA_cm2']
                js.append(current); cds.append(diag)
            timings[mode].append(time.perf_counter()-tstart); currents[mode]=js; control_diagnostics[mode]=cds
            print(f'Benchmark {rep+1}/3 {mode}: {timings[mode][-1]:.3f}s',flush=True)
    med={key:float(np.median(value)) for key,value in timings.items()}
    ref=np.array(currents[modes[0]])
    for mode in modes[1:]:
        assert np.max(abs(np.array(currents[mode])-ref)/abs(ref))<2e-3
    report['benchmark']=dict(voltages=bvs,repeat_seconds=timings,median_seconds=med,currents_mA_cm2=currents,control_diagnostics=control_diagnostics,
         measured_python_speedup=med[modes[0]]/med[modes[-1]],
         caveat='Same stabilized model, scaled tolerances, BDF and vectorized RHS in all modes; not a MATLAB benchmark. Fixed-time controls may fail the strict residual criterion despite matching currents; see control_diagnostics. Full mode includes exciton transients; DC modes omit them.')
    (out/'validation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['benchmark'],indent=2),flush=True)
    print('All validation checks PASSED',flush=True)


if __name__=='__main__':
    main()