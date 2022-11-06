import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

# TODO Flesh out docstrings
# TODO Check optional argument default values


def Q(r, q0):
    """Returns q=sqrt(q0^2 + r^2)."""
    return np.sqrt(q0**2 + r**2)

def Qd(r, q0):
    """Returns derivative of q=sqrt(q0^2 + r^2)."""
    return r / Q(r, q0)

def Qdd(r, q0):
    """Returns second derivative of q=sqrt(q0^2 + r^2)."""
    return q0**2 / Q(r, q0)**3

def f_event(r, y, q0, flux):
    """Stops integration if f > 1/q (appropriate for AdS)."""
    f = y[0]
    return f - 3/Q(r, q0)
f_event.terminal = True


###################################################################################################
#
## Methods for massive type IIA on S3xS3
#
###################################################################################################

def V_S3S3(u, φ):
    """Scalar potential V(u,φ) for S3xS3."""
    return -12*np.exp(-4*u) + 5*np.exp(-9*u-φ/2) + np.exp(-3*u+5*φ/2)

def dVdu_S3S3(u, φ):
    """u-derivative of scalar potential V(u,φ) for S3xS3."""
    return 48*np.exp(-4*u) - 45*np.exp(-9*u-φ/2) - 3*np.exp(-3*u+5*φ/2)

def dVdφ_S3S3(u, φ):
    """φ-derivative of scalar potential V(u,φ) for S3xS3."""
    return -(5/2)*np.exp(-9*u-φ/2) + (5/2)*np.exp(-3*u+5*φ/2)

def ODEs_S3S3(r, y, q0, f4sqr):
    """ODEs for f, u, φ and h."""

    # Unpack
    f, u, ud, φ, φd, h = y

    q = Q(r, q0)
    qd = Qd(r, q0)
    qdd = Qdd(r, q0)
    
    # By definition
    u_d = ud
    φ_d = φd
    
    # Equations of motion
    prefactor = qdd/qd - qd/q - f**2 / (q*qd) * (2 - 1/2 * q**2 * V_S3S3(u, φ))
    
    f_d  = (prefactor + 3*qd/q) * f
    ud_d = prefactor*ud + f**2 * ((1/12)*dVdu_S3S3(u, φ) + f4sqr*np.exp(3*u-φ/2)/(8*q**6))
    φd_d = prefactor*φd + f**2 * (       dVdφ_S3S3(u, φ) - f4sqr*np.exp(3*u-φ/2)/(4*q**6))
    h_d  = f/q**3
    
    # Return derivatives of f, u, ud, φ, φd and h
    return f_d, u_d, ud_d, φ_d, φd_d, h_d

def solve_S3S3(q0, rmax, u0, φ0, rmin=10**-8, nr=1000):
    """Solves ODEs for S3xS3 out to r=rmax for given q0 and initial conditions u0,φ0."""

    # Initial condition for f0 and the value of f4^2 are set by u0 and φ0
    # f0m2 = f0**(-2)
    f0m2 = 2 - 1/2 * q0**2 * V_S3S3(u0, φ0)
    f4sqr = 4*np.exp(-3*u0+φ0/2) * q0**4 * (1 + f0m2)
    
    if f0m2 < 0:
        # Regular solutions must have f0^2 > 0
        return [f0m2]

    f0 = f0m2**(-1/2)

    # Use series solutions (e.g. u = u0 + 1/2*udd0 * r^2 + ...)
    # to get initial conditions at r=rmin, where 0 < rmin << q0
    udd0 = f0**2 * ( 3/(2*q0**2) - (1/4)*V_S3S3(u0, φ0) + (1/12)*dVdu_S3S3(u0, φ0))
    φdd0 = f0**2 * (-3/q0**2     + (1/2)*V_S3S3(u0, φ0) +        dVdφ_S3S3(u0, φ0))
    fdd0 = (f0**3 * q0**2 / 8) * (8/q0**4 + udd0*dVdu_S3S3(u0, φ0) + φdd0*dVdφ_S3S3(u0, φ0))

    f_start = f0 + (1/2)*fdd0*rmin**2

    u_start = u0 + (1/2)*udd0*rmin**2
    ud_start = udd0 * rmin

    φ_start = φ0 + (1/2)*φdd0*rmin**2
    φd_start = φdd0 * rmin

    h_start = rmin * f0 / q0**3

    # Initial conditions and constants
    y0 = (f_start, u_start, ud_start, φ_start, φd_start, h_start)
    args = (q0, f4sqr)

    soln = solve_ivp(ODEs_S3S3, (rmin, rmax),
                     y0=y0,
                     args=args,
                     events=(f_event),
                    #  t_eval=np.linspace(rmin, rmax, nr),
                     t_eval=np.geomspace(rmin, rmax, nr),
                     rtol=10**-8,
                     method='RK45'
                    )

    # Return (r, f, u, ud, φ, φd, h, f4sqr)
    return [soln.t, *soln.y, f4sqr]

def objective_S3S3(uφ0, q0, rmax, rmin=10**-8, nr=1000, display_progress=False):
    """Objective function to be minimized during shooting method."""

    soln = solve_S3S3(q0, rmax, *uφ0, rmin)

    if len(soln) == 1:
        # Invalid (f0**2<0): return (const.)+|f0| to drive towards f0**2>0
        if display_progress:
            print('{:24.20f} {:24.20f}\t invalid'.format(*uφ0))

        value = 10**8 + abs(soln[0])
    else:
        # Unpack solution
        r, f, u, ud, φ, φd, h, f4sqr = soln
        
        Δ1 = (3/2) + np.sqrt((3/2)**2 + 6)      # Conformal dimension for light mode

        value = 1 + 10**3 * (rmax/r[-1] - 1)

        if r[-1] == rmax:
            value -= 1/(1 + ((u[-1] + 0.1*φ[-1]) + r[-1]*(ud[-1] + 0.1*φd[-1])/6)**2 \
                        + (φ[-1] + r[-1]*φd[-1]/Δ1)**2)

            if display_progress:
                print('{:24.20f} {:24.20f}\t   good!: {:4.2f}\t{:46.40f}'.format(*uφ0, r[-1], value))

        else:
            value += u[-1]**2 + φ[-1]**2

            if display_progress:
                print('{:24.20f} {:24.20f}\tsingular: {:4.2f}\t{:46.40f}'.format(*uφ0, r[-1], value))

    return value

def paramScan_S3S3(q0, rmax, u0_bounds, φ0_bounds, u0_steps, φ0_steps):
    """Displays the objective function for a grid of u0,φ0 values."""

    u0_list = np.linspace(u0_bounds[0], u0_bounds[1], u0_steps)
    φ0_list = np.linspace(φ0_bounds[0], φ0_bounds[1], φ0_steps)

    data = []

    for u0, φ0 in tqdm(product(u0_list, φ0_list), total=len(u0_list)*len(φ0_list)):
        quality = objective_S3S3([u0, φ0], q0, rmax)
        data.append([u0, φ0, quality])
    
    data = np.asarray(data).T

    plotData = data[:, data[2] < 1000]

    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    im = plt.tricontourf(plotData[0], plotData[1], plotData[2],
                         levels=20,
                         cmap='cet_CET_L12_r',
                         origin='lower',
                         extent=(u0_list[0], u0_list[-1], φ0_list[0], φ0_list[-1])
                        )

    plt.xlim(u0_list[0], u0_list[-1])
    plt.ylim(φ0_list[0], φ0_list[-1])

    plt.xlabel('$u_0$')
    plt.ylabel('$\phi_0$')

    plt.colorbar(im)

    plt.show()

def wormhole_S3S3(q0, rmax, rmax_steps=3, uφ0=[0,0], xatol=10**-8, nr=1000, display_progress=False):
    """Return optimal solution out to r=rmax for q0 and with r=0 boundary conditions found using shooting method."""

    print('\nPerforming shooting method for q0 = {} out to r = {} ...'.format(q0, rmax))

    # Optimize several times, increasing rmax and updating (u0,φ0) at each step
    rmax_list = np.geomspace(q0, rmax, rmax_steps)
    uφ0_best = uφ0

    for ii, rmax_ii in enumerate(rmax_list):
        
        # Use lower precision except for final optimization
        xatol_ii = 10**-4
        if ii+1 == rmax_steps:
            xatol_ii = xatol

        print('    rmax = {:.4f} with xatol = {}'.format(rmax_ii, xatol_ii))

        # Shooting method: determine u0,φ0 to match boundary conditions/scaling solutions at r>>q0
        opt = minimize(lambda uφ0: objective_S3S3(uφ0, q0, rmax_ii, display_progress=display_progress),
                       x0=uφ0_best,
                       method='Nelder-Mead',
                       options={'maxfev': 1000,
                                'xatol': xatol_ii,
                               }
                      )
        uφ0_best = opt.x

        # Print results
        print('{:>14} : {}'.format('f_eval', opt.nfev))
        print('{:>14} : {:+.10g}'.format('u0', uφ0_best[0]))
        print('{:>14} : {:+.10g}'.format('φ0', uφ0_best[1]))
        print('{:>14} : {:.10g}'.format('val', opt.fun))

    # Get numerical solution for optimial initial conditions
    soln = solve_S3S3(q0, rmax, *uφ0_best, nr=nr)

    return symmetrize_S3S3(soln)

def symmetrize_S3S3(soln):
    """Extend solutions from 0 < r < rmax to -rmax < r < rmax."""

    r, f, u, ud, φ, φd, h, f4sqr = soln
    r[0] = 0

    # Make functions even/odd about r=0
    r = np.append(-r[-1:0:-1], [r])
    f = np.append( f[-1:0:-1], [f])
    u = np.append( u[-1:0:-1], [u])
    φ = np.append( φ[-1:0:-1], [φ])
    h = np.append(-h[-1:0:-1], [h])

    ud = np.append(-ud[-1:0:-1], [ud])
    φd = np.append(-φd[-1:0:-1], [φd])

    return r, f, u, ud, φ, φd, h, f4sqr

def massless_approx_S3S3(q0):
    
    # Taking h(0) = 0, first compute h(inf)
    h_inf, err = quad(lambda x: q0**-2 * x**2 * (q0**2 + x**2 - (1+q0**2)*x**6)**(-1/2),
                      0, 1
                     )

    # φ = -2u has profile exp(φ) = cos(1/2 * sqrt(-c)*h) / cos(1/2 * sqrt(-c)*hinf),
    #   so φ(0) = -log[cos(1/2 * sqrt(-c)*hinf)]
    c_abs = 12*q0**4 * (1+q0**2)
    f4 = np.sqrt(c_abs) / np.cos(0.5 * np.sqrt(c_abs) * h_inf)
    φ0 = -np.log(np.cos(0.5 * np.sqrt(c_abs) * h_inf))
    u0 = -0.5*φ0

    return u0, φ0, f4

###################################################################################################
#
## Methods for type IIB on T(1,1)
#
###################################################################################################


def V_T11(u, v):
    """Scalar potential V(u,v) for T11."""
    return 2*np.exp(-8/3*(4*u+v)) * (2*np.exp(4*u+4*v) - 12*np.exp(6*u+2*v) + 4)

def dVdu_T11(u, v):
    """u-derivative of scalar potential V(u,v) for T11."""
    return -(16/3)*np.exp(-8/3*(4*u+v)) * (5*np.exp(4*u+4*v) - 21*np.exp(6*u+2*v) + 16)

def dVdv_T11(u, v):
    """v-derivative of scalar potential V(u,v) for T11."""
    return (16/3)*np.exp(-8/3*(4*u+v)) * (np.exp(4*u+4*v) + 3*np.exp(6*u+2*v) - 4)

def ODEs_T11(r, y, q0, f3sqr):
    """ODEs for f, u, v, φ and h."""

    # Unpack
    f, u, ud, v, vd, φ, φd, h = y

    q = Q(r, q0)
    qd = Qd(r, q0)
    qdd = Qdd(r, q0)

    # By definition
    u_d = ud
    v_d = vd
    φ_d = φd

    # Equations of motion
    prefactor = qdd/qd - qd/q - f**2/(q*qd) * (3 - 1/3 * q**2 * V_T11(u, v))
    
    f_d  = (prefactor + 4*qd/q) * f
    ud_d = prefactor*ud + (f**2 / 16) * ( dVdu_T11(u, v) -   dVdv_T11(u, v) + 2*f3sqr*np.exp(4*u-φ)/q**8)
    vd_d = prefactor*vd + (f**2 / 16) * (-dVdu_T11(u, v) + 7*dVdv_T11(u, v) - 2*f3sqr*np.exp(4*u-φ)/q**8)
    φd_d = prefactor*φd - f3sqr*np.exp(4*u-φ) * f**2/(2*q**8)

    h_d = f/q**4

    # Return derivatives of w, u, ud, v, vd, φ, φd and h
    return f_d, u_d, ud_d, v_d, vd_d, φ_d, φd_d, h_d

def solve_T11(q0, rmax, u0, v0, φ0, rmin=10**-8, nr=1000):
    """Solves ODEs for T11 out to r=rmax for given q0 and initial conditions u0,v0,φ0."""

    # Initial condition for f0 and the value of f3^2 are set by u0, v0 and φ0
    # f0m2 = f0**(-2)
    f0m2 = 3 - 1/3 * q0**2 * V_T11(u0, v0)
    f3sqr = 6*np.exp(-4*u0+φ0) * q0**6 * (1 + f0m2)
    
    if f0m2 < 0:
        # Regular solutions must have f0**2>0
        return [f0m2]

    f0 = f0m2**(-1/2)

    # Use series solutions (e.g. u = u0 + 1/2*udd0 * r^2 + ...)
    # to get initial conditions at 0 < rmin << q0
    udd0 = (f0**2 / 16) * ( dVdu_T11(u0, v0) -   dVdv_T11(u0, v0) + 2*f3sqr*np.exp(4*u0-φ0)/q0**8)
    vdd0 = (f0**2 / 16) * (-dVdu_T11(u0, v0) + 7*dVdv_T11(u0, v0) - 2*f3sqr*np.exp(4*u0-φ0)/q0**8)
    φdd0 = -f3sqr*np.exp(4*u0-φ0) * f0**2/(2*q0**8)
    fdd0 = (f0**3 * q0**2 / 12) * (6*(3+f0**(-2))/q0**4 + udd0*dVdu_T11(u0, v0) + vdd0*dVdv_T11(u0, v0))
    
    f_start = f0 + (1/2)*fdd0*rmin**2

    u_start = u0 + (1/2)*udd0*rmin**2
    ud_start = udd0 * rmin
    
    v_start = v0 + (1/2)*vdd0*rmin**2
    vd_start = vdd0 * rmin

    φ_start = φ0 + (1/2)*φdd0*rmin**2
    φd_start = φdd0 * rmin

    h_start = rmin * f0 / q0**4

    # Initial conditions and constants
    y0 = (f_start, u_start, ud_start, v_start, vd_start, φ_start, φd_start, h_start)
    args = (q0, f3sqr)

    soln = solve_ivp(ODEs_T11, (rmin, rmax),
                     y0=y0,
                     args=args,
                     events=(f_event),
                     t_eval=np.geomspace(rmin, rmax, nr),
                     rtol=10**-8,
                     method='RK45'
                    )

    # Return (r, f, u, ud, v, vd, φ, φd, h, f4sqr)
    return [soln.t, *soln.y, f3sqr]

def objective_T11(uv0, φ0, q0, rmax, rmin=10**-8, nr=1000, display_progress=False):
    """Objective function to be minimized during shooting method."""
    
    soln = solve_T11(q0, rmax, *uv0, φ0, rmin)

    if len(soln) == 1:
        # Invlid (f0**2<0): return (const.)+|f0| to drive towards f0**2>0
        if display_progress:
            print('{:24.20f} {:24.20f}\tinvalid'.format(*uv0))

        value = 10**8 + abs(soln[0])
    else:
        # Unpack solution
        r, f, u, ud, v, vd, φ, φd, h, f3sqr = soln

        Δ1 = (4/2) + np.sqrt((4/2)**2 + 12)     # Conformal dimension for light mode

        value = 1 + 10**3 * (rmax/r[-1] - 1)

        if r[-1] == rmax:
            value -= 1/(1 + ((u[-1] + 0.25*v[-1]) + r[-1]*(ud[-1] + 0.25*vd[-1])/8)**2 \
                        + (v[-1] + r[-1]*vd[-1]/Δ1)**2)
            # value -= 1/(1 + (u[-1]**2 + (v[-1] + r[-1]*vd[-1]/Δ1)**2))
            # value -= 1/(1 + ((u[-1] + 0.25*v[-1])**2 + (v[-1] + r[-1]*vd[-1]/Δ1)**2))
            if display_progress:
                print('{:24.20f} {:24.20f}\t   good!: {:4.2f}\t{:46.40f}'.format(*uv0, r[-1], value))

        else:
            value += u[-1]**2 + v[-1]**2

            if display_progress:
                print('{:24.20f} {:24.20f}\tsingular: {:4.2f}\t{:46.40f}'.format(*uv0, r[-1], value))

    return value

def paramScan_T11(q0, rmax, u0_bounds, v0_bounds, u0_steps, v0_steps):
    """Displays the objective function for a grid of u0,v0 values."""

    u0_list = np.linspace(u0_bounds[0], u0_bounds[1], u0_steps)
    v0_list = np.linspace(v0_bounds[0], v0_bounds[1], v0_steps)

    data = []

    for u0, v0 in tqdm(product(u0_list, v0_list), total=len(u0_list)*len(v0_list)):
        quality = objective_T11([u0, v0], 0, q0, rmax)
        data.append([u0, v0, quality])
    
    data = np.asarray(data).T

    plotData = data[:, data[2] < 1000]

    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    im = plt.tricontourf(plotData[0], plotData[1], plotData[2],
                         levels=20,
                         cmap='cet_CET_L12_r',
                         origin='lower',
                         extent=(u0_list[0], u0_list[-1], v0_list[0], v0_list[-1])
                        )

    plt.xlim(u0_list[0], u0_list[-1])
    plt.ylim(v0_list[0], v0_list[-1])

    plt.xlabel('$u_0$')
    plt.ylabel('$v_0$')

    plt.colorbar(im)

    plt.show()

def wormhole_T11(q0, rmax, rmax_steps=3, uv0=[0,0], xatol=10**-8, nr=1000, display_progress=False):
    """Return optimal solution out to r=rmax for q0 and with r=0 boundary conditions found using shooting method."""

    print('\nPerforming shooting method for q0 = {} out to r = {} ...'.format(q0, rmax))

    # Optimize several times, increasing rmax and updating (u0,v0) at each step
    rmax_list = np.geomspace(q0, rmax, rmax_steps)
    uv0_best = uv0

    for ii, rmax_ii in enumerate(rmax_list):
        
        # Use lower precision except for final optimization
        xatol_ii = 10**-4
        if ii+1 == rmax_steps:
            xatol_ii = xatol

        print('    rmax = {:.4f} with xatol = {}'.format(rmax_ii, xatol_ii))

        # Shooting method: determine u0,v0 to match boundary conditions/scaling solutions at r>>q0
        # Set φ0 = 0 for optimizing u0,v0 since can always absorb constant into f3^2 at the end
        opt = minimize(lambda uv0: objective_T11(uv0, 0, q0, rmax_ii, display_progress=display_progress),
                       x0=uv0_best,
                       method='Nelder-Mead',
                       options={'maxfev': 1000,
                                'xatol': xatol_ii,
                               }
                      )
        uv0_best = opt.x

        # Print results
        print('{:>14} : {}'.format('f_eval', opt.nfev))
        print('{:>14} : {:+.10g}'.format('u0', uv0_best[0]))
        print('{:>14} : {:+.10g}'.format('v0', uv0_best[1]))
        print('{:>14} : {:.10g}'.format('val', opt.fun))

    # Get numerical solution for optimial initial conditions
    soln = solve_T11(q0, rmax, *uv0_best, 0, nr=nr)
    r, f, u, ud, v, vd, φ, φd, h, f3sqr = soln

    # Shift φ and rescale flux so that φ(inf) -> 0
    # This leverages the fact that only the combination f^2 * exp(-φ) appears in the equations of motion
    φinf = φ[-1] + r[-1]*φd[-1]/4
    soln[6] -= φinf
    soln[9] *= np.exp(-φinf/2)

    return symmetrize_T11(soln)

def symmetrize_T11(soln):
    """Extend solutions from 0 < r < rmax to -rmax < r < rmax."""

    r, f, u, ud, v, vd, φ, φd, h, f3sqr = soln
    r[0] = 0

    # Make (anti-)symmetric about r=0
    r = np.append(-r[-1:0:-1], [r])
    f = np.append( f[-1:0:-1], [f])
    u = np.append( u[-1:0:-1], [u])
    v = np.append( v[-1:0:-1], [v])
    φ = np.append( φ[-1:0:-1], [φ])
    h = np.append(-h[-1:0:-1], [h])

    ud = np.append(-ud[-1:0:-1], [ud])
    vd = np.append(-vd[-1:0:-1], [vd])
    φd = np.append(-φd[-1:0:-1], [φd])

    return r, f, u, ud, v, vd, φ, φd, h, f3sqr

def massless_approx_T11(q0):
    
    # Taking h(0) = 0, first compute h(inf)
    h_inf, err = quad(lambda x: q0**-3 * x**3 * (q0**2 + x**2 - (1+q0**2)*x**8)**(-1/2),
                      0, 1
                     )

    # φ = -4u = 4v has profile exp(φ) = cos(sqrt(-c/2)*h) / cos(sqrt(-c/2)*hinf),
    #   so φ(0) = -log[cos(sqrt(-c/2)*hinf)]
    c_abs = 24*q0**6 * (1+q0**2)
    f3 = np.sqrt(c_abs) / np.cos(np.sqrt(c_abs/2) * h_inf)
    φ0 = -np.log(np.cos(np.sqrt(c_abs/2) * h_inf))
    u0 = -φ0/4
    v0 = φ0/4

    return u0, v0, φ0, f3, h_inf

def frozen_approx_T11(q0):

    # Taking h(0) = 0, first compute h(inf)
    h_inf, err = quad(lambda x: q0**-3 * x**3 * (q0**2 + x**2 - (1+q0**2)*x**8)**(-1/2),
                      0, 1
                     )

    # u=v=0 and exp(φ/2) = cos(1/2 * sqrt(-c)*h) / cos(1/2 * sqrt(-c)*hinf),
    #   so φ(0) = -2*log[cos(1/2 * sqrt(-c)*hinf)]
    c_abs = 24*q0**6 * (1 + q0**2)
    f3 = np.sqrt(c_abs) / np.cos(0.5*np.sqrt(c_abs) * h_inf)
    φ0 = -2*np.log(np.cos(0.5*np.sqrt(c_abs) * h_inf))

    return 0, 0, φ0, f3, h_inf

def ODE_φ(r, y, q0, f3sqr):

    φ, φd = y

    q = Q(r, q0)
    qd = Qd(r, q0)
    qdd = Qdd(r, q0)

    φ_d = φd

    f = qd / np.sqrt(1 + q**2 - q0**6 * (1+q0**2)/q**6)

    # Equation of motion
    prefactor = qdd/qd - qd/q - f**2/(q*qd) * (3 + 4*q**2)
    
    φd_d = prefactor*φd - f3sqr*np.exp(-φ) * f**2/(2*q**8)

    return φ_d, φd_d


def wormhole_frozen_T11(q0, rmax, nr=1000):

    rmin = 10**-8

    r = np.geomspace(rmin, rmax, nr)

    q = Q(r, q0)
    qd = Qd(r, q0)

    c_abs = 24*q0**6 * (1 + q0**2)

    f = qd / np.sqrt(1 + q**2 - q0**6 * (1+q0**2)/q**6)
    f[0] = 1 / np.sqrt(3 + 4*q0**2)

    u0, v0, φ0, f3, h_inf = frozen_approx_T11(q0)

    f3 = np.sqrt(c_abs) / np.cos(0.5*np.sqrt(c_abs) * h_inf)


    soln = solve_ivp(ODE_φ, (rmin, rmax),
                     y0=[φ0, 0],
                     args=[q0, f3**2],
                     t_eval=r,
                     rtol=10**-12,
                     method='RK45'
                    )
    φ, φd = soln.y

    φ -= φ[-1] + r[-1]*φd[-1]/4

    u = 0*r
    ud = 0*r
    v = 0*r
    vd = 0*r

    soln = [r, f, u, ud, v, vd, φ, φd, 0*r, f3**2]

    return symmetrize_T11(soln)
