import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, curve_fit
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
    """Stops integration if f >> 1/q (appropriate for AdS)."""
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

def ODEs_S3S3(r, y, q0, flux4):
    """ODEs describing evolution of y=(f, u, u', φ, φ', h) for S3xS3."""

    # Unpack y
    f, u, ud, φ, φd, h = y

    # Get q(r) and its derivatives
    q   =   Q(r, q0)
    qd  =  Qd(r, q0)
    qdd = Qdd(r, q0)
    
    # By definition
    u_d = ud
    φ_d = φd
    
    # Equations of motion
    prefactor = qdd/qd - qd/q - f**2 / (q*qd) * (2 - (1/2)*q**2 * V_S3S3(u, φ))
    
    f_d  = (prefactor + 3*qd/q) * f
    ud_d = prefactor*ud + f**2 * ((1/12)*dVdu_S3S3(u, φ) + (1/8)*flux4**2 * np.exp(3*u-φ/2)/q**6)
    φd_d = prefactor*φd + f**2 * (       dVdφ_S3S3(u, φ) - (1/4)*flux4**2 * np.exp(3*u-φ/2)/q**6)
    h_d  = f/q**3
    
    # Return derivatives of f, u, ud, φ, φd and h
    return f_d, u_d, ud_d, φ_d, φd_d, h_d

def solve_S3S3(q0, u0, φ0, rmax, rmin=10**-6, nr=1000):
    """Solves ODEs for S3xS3 out to r=`rmax` for given `q0` and initial conditions (`u0`,`φ0`).

    Parameters
    ----------
    q0 : float
        Wormhole size in AdS units.
    u0 : float
        Initial condition, `u`(r=0).
    φ0 : float
        Initial condition, `φ`(r=0).
    rmax : float
        End point of integration.
    rmin : float (optional)
        Start point of integration (`rmin` << `q0`).
    nr : int (optional)
        Number of r > 0 values at which to evaluate the solutions.
    
    Returns
    -------
    soln : array
        If the initial conditions are invalid, [f0m2], a measure of how far from being admissible the ICs are.
        Otherwise, the numerical wormhole solution, (r, f, u, ud, φ, φd, h, flux4),
        where all but flux4 are arrays of length `nr`.

    """

    # Initial condition for f0 and the value of flux4^2 are set by u0 and φ0 (f0m2 := f0**(-2))
    f0m2 = 2 - (1/2)*q0**2 * V_S3S3(u0, φ0)
    flux4sqr = 4*np.exp(-3*u0+φ0/2) * q0**4 * (1 + f0m2)
    
    # Regular solutions must have f0^2 > 0
    if f0m2 < 0:
        # Return value of f0m2 for use in shooting method
        return [f0m2]

    f0 = f0m2**(-1/2)
    flux4 = np.sqrt(flux4sqr)

    # Use series solutions (e.g. u = u0 + 1/2*udd0 * r^2 + ...)
    # to get initial conditions at r=rmin, where 0 < rmin << q0
    udd0 = f0**2 * ( 3/(2*q0**2) - (1/4)*V_S3S3(u0, φ0) + (1/12)*dVdu_S3S3(u0, φ0))
    φdd0 = f0**2 * (-3/q0**2     + (1/2)*V_S3S3(u0, φ0) +        dVdφ_S3S3(u0, φ0))
    fdd0 = (f0**3 * q0**2 / 8) * (8/q0**4 + udd0*dVdu_S3S3(u0, φ0) + φdd0*dVdφ_S3S3(u0, φ0))

    f_start = f0 + (1/2)*fdd0 * rmin**2
    u_start = u0 + (1/2)*udd0 * rmin**2
    φ_start = φ0 + (1/2)*φdd0 * rmin**2

    ud_start = udd0 * rmin
    φd_start = φdd0 * rmin

    h_start = rmin * f0 / q0**3

    # Initial conditions and constants
    y0 = (f_start, u_start, ud_start, φ_start, φd_start, h_start)
    args = (q0, flux4)

    soln = solve_ivp(ODEs_S3S3, (rmin, rmax),
                     y0=y0,
                     args=args,
                     events=(f_event),  # halt if f gets too large
                     t_eval=np.geomspace(rmin, rmax, nr),
                     rtol=10**-8,
                     method='RK45'
                    )

    # Return (r, f, u, ud, φ, φd, h, flux4)
    return [soln.t, *soln.y, flux4]

def objective_S3S3(uφ0, q0, rmax, rmin=10**-6, display=None):
    """Objective function to be minimized during S3xS3 shooting method."""

    # Solve EoMs for given ICs
    soln = solve_S3S3(q0, *uφ0, rmax, rmin)

    if len(soln) == 1:
        # ICs are invalid (i.e. f0^(-2) < 0): set value to drive towards admissible ICs
        value = 10**8 + abs(soln[0])
    else:
        # Unpack solution
        r, f, u, ud, φ, φd, h, flux4 = soln

        if r[-1] == rmax:
            # Regular solution out to r=rmax: reward if u,φ are approaching zero

            Δ1 = (3/2) + np.sqrt((3/2)**2 + 6)  # Conformal dimension for light mode

            # Estimate the values of φ and u+0.1φ (the heavy mode) at r=infty using
            # the expected power-law solutions, φ ~ r^(-Δ1) and (u+0.1φ) ~ r^(-6)
            φinf_est = φ[-1] + r[-1]*φd[-1]/Δ1
            uφinf_est = (u[-1] + 0.1*φ[-1]) + r[-1]*(ud[-1] + 0.1*φd[-1])/6     # TODO: Change to just u?

            # Reward when the extrapolated values φ(infty) and (u+0.1φ)(infty) are small
            value = 1 - 1/(1 + φinf_est**2 + uφinf_est**2)
        else:
            # Singular solution: penalize to drive towards nonsingular u,φ
            value = 1 + 10**3 * (rmax/r[-1] - 1) + u[-1]**2 + φ[-1]**2

    # Print ICs and objective function
    if display is 'progress':
        print('{:19.15f} {:19.15f} \t {:8.4f} {:46.40f}'.format(*uφ0, r[-1], value))

    return value

def wormhole_S3S3(q0, rmax, rmin=10**-6, nr=1000, xatol=10**-10, display=None):
    """Return optimal solution out to r=`rmax` for `q0` and with r=0 boundary conditions found using shooting method.
    
    Parameters
    ----------
    q0 : float
        Wormhole size in AdS units.
    rmax : float
        End point of integration.
    rmin : float (optional)
        Start point of integration (`rmin` << `q0`).
    nr : int (optional)
        Number of r > 0 values at which to evaluate the solutions.
    xatol : float (optional)
        Option for Nelder-Mead method.
    display : [None, 'quiet', 'summary', 'progress'] (optional)
        Controls what information about the shooting method is printed.
            None (default) : nothing printed
            'quiet'        : single line giving parameters and final value of objective function
            'summary'      : summary info for final optimization, including optimal (u0, φ0)
            'progress'     : (u0, φ0) and objective function for every step of the optimization

    Returns
    -------
    soln : array
        The numerical wormhole solution, (r, f, u, ud, φ, φd, h, flux4), where all but flux4
        are arrays of length 2*`nr`-1 (being symmetrized on the domain -`rmax` < r < `rmax`).
    
    """

    if display is not None:
        print('Optimizing (u0, φ0) for (q0, rmax) = ({:.4f}, {:.4f})'.format(q0, rmax), end='')

    # Perform shooting method a few times:
    #  - If q0 < 1 it helps to first integrate out to r=q0 with low precision
    #  - Integrate out to r=max(1, q0) where AdS scaling solutions begin with low precision
    #  - Finally, integrate out to r=rmax at higher precision
    rmax_list = [max(1, q0), rmax]
    xatol_list = [0.1, xatol]

    if q0 < 1:
        rmax_list.insert(0, q0)
        xatol_list.insert(0, 0.1)

    # Keep track of best ICs
    uφ0_best = [0, 0]

    for rm, xat in zip(rmax_list, xatol_list):
        # Shooting method: optimize (u0,φ0) to match AdS BCs
        opt = minimize(lambda uφ0: objective_S3S3(uφ0, q0, rm, rmin, display),
                       x0=uφ0_best,
                       method='Nelder-Mead',
                       options={'maxfev': 1000, 'xatol': xat}
                      )
        # Record best (u0,φ0) found
        uφ0_best = opt.x

    # Print results
    if display is 'quiet':
        print('\tDONE: value = {:.4g}'.format(opt.fun))

    elif display is 'summary':
        print('\n{:>14} : {}'.format('success', opt.success))
        print('{:>14} : {}'.format('f_eval', opt.nfev))
        print('{:>14} : {:+.10g}'.format('u0', uφ0_best[0]))
        print('{:>14} : {:+.10g}'.format('v0', uφ0_best[1]))
        print('{:>14} : {:.10g}'.format('value', opt.fun))

    # Get numerical solution for optimial initial conditions
    soln = solve_S3S3(q0, *uφ0_best, rmax, rmin, nr)

    # Return symmetrized solution on -rmax < r < rmax
    return symmetrize_S3S3(soln)

def symmetrize_S3S3(soln):
    """Extend S3xS3 solutions from 0 < r < rmax to -rmax < r < rmax."""

    r, f, u, ud, φ, φd, h, flux4 = soln
    r[0] = 0

    # Make functions even/odd about r=0
    r = np.append(-r[-1:0:-1], [r])
    f = np.append( f[-1:0:-1], [f])
    u = np.append( u[-1:0:-1], [u])
    φ = np.append( φ[-1:0:-1], [φ])
    h = np.append(-h[-1:0:-1], [h])

    ud = np.append(-ud[-1:0:-1], [ud])
    φd = np.append(-φd[-1:0:-1], [φd])

    return r, f, u, ud, φ, φd, h, flux4

def massless_approx_S3S3(q0):
    """Returns (u0,φ0) and flux for the massless "approximation" (assuming u,φ vanish at infinity)."""
    
    # Taking h(0) = 0, first compute h(inf)
    h_inf, err = quad(lambda x: q0**-2 * x**2 * (q0**2 + x**2 - (1+q0**2)*x**6)**(-1/2),
                      0, 1
                     )

    # φ = -2u has profile exp(φ) = cos(1/2 * sqrt(-c)*h) / cos(1/2 * sqrt(-c)*hinf),
    #   so φ(0) = -log[cos(1/2 * sqrt(-c)*hinf)]
    c_abs = 12*q0**4 * (1+q0**2)
    flux4 = np.sqrt(c_abs) / np.cos(0.5 * np.sqrt(c_abs) * h_inf)
    φ0 = -np.log(np.cos(0.5 * np.sqrt(c_abs) * h_inf))
    u0 = -0.5*φ0

    return u0, φ0, flux4

###################################################################################################
#
## Methods for type IIB on T(1,1)
#
###################################################################################################


def masslessScalarFit(r, cinf, c4, c6):
    return cinf + c4/r**4 + c6/r**6

def masslessScalarFit_alt(r, c4, c6):
    return -4*c4 - 6*c6/r**2

def V_T11(u, v):
    """Scalar potential V(u,v) for T11."""
    return 2*np.exp(-8/3*(4*u+v)) * (2*np.exp(4*u+4*v) - 12*np.exp(6*u+2*v) + 4)

def dVdu_T11(u, v):
    """u-derivative of scalar potential V(u,v) for T11."""
    return -(16/3)*np.exp(-8/3*(4*u+v)) * (5*np.exp(4*u+4*v) - 21*np.exp(6*u+2*v) + 16)

def dVdv_T11(u, v):
    """v-derivative of scalar potential V(u,v) for T11."""
    return (16/3)*np.exp(-8/3*(4*u+v)) * (np.exp(4*u+4*v) + 3*np.exp(6*u+2*v) - 4)

def ODEs_T11(r, y, q0, flux2):
    """ODEs describing evolution of y=(f, u, u', v, v', φ, φ', χ, χ', h) for T11."""

    # Unpack
    f, u, ud, v, vd, φ, φd, χ, χd, h = y

    # Get q(r) and its derivatives
    q   =   Q(r, q0)
    qd  =  Qd(r, q0)
    qdd = Qdd(r, q0)

    # By definition
    u_d = ud
    v_d = vd
    φ_d = φd
    χ_d = χd

    # Equations of motion
    prefactor = qdd/qd - qd/q - f**2/(q*qd) * (3 - (1/3)*q**2 * V_T11(u, v))
    
    f_d  = (prefactor + 4*qd/q) * f
    ud_d = prefactor*ud + (f**2 / 16) * ( dVdu_T11(u, v) -   dVdv_T11(u, v)) \
           - (1/8)*flux2**2 * np.exp(4*u+φ) * (χ**2 - np.exp(-2*φ)) * f**2/q**8
    vd_d = prefactor*vd + (f**2 / 16) * (-dVdu_T11(u, v) + 7*dVdv_T11(u, v)) \
           + (1/8)*flux2**2 * np.exp(4*u+φ) * (χ**2 - np.exp(-2*φ)) * f**2/q**8
    φd_d = prefactor*φd - np.exp(2*φ)*χd**2 - (1/2)*flux2**2 * np.exp(4*u+φ) * (χ**2 + np.exp(-2*φ)) * f**2/q**8
    χd_d = prefactor*χd - 2*φd*χd + flux2**2 * np.exp(4*u-φ) * χ * f**2/q**8

    h_d = f/q**4

    # Return derivatives
    return f_d, u_d, ud_d, v_d, vd_d, φ_d, φd_d, χ_d, χd_d, h_d

def solve_T11(q0, u0, v0, φ0, χ1, rmax, rmin=10**-6, nr=1000):
    """Solves ODEs for T11 out to r=`rmax` for given `q0` and initial conditions (`u0`,`v0`,`φ0`,`χ1`).

    Parameters
    ----------
    q0 : float
        Wormhole size in AdS units.
    u0 : float
        Initial condition, `u`(r=0).
    v0 : float
        Initial condition, `v`(r=0).
    φ0 : float
        Initial condition, `φ`(r=0).
    χ1 : float
        Initial condition, d`χ`/dr(r=0).
    rmax : float
        End point of integration.
    rmin : float (optional)
        Start point of integration (`rmin` << `q0`).
    nr : int (optional)
        Number of r > 0 values at which to evaluate the solutions.
    
    Returns
    -------
    soln : array
        If the initial conditions are invalid, [f0m2, flux2sqr], a measure of how far from being admissible the ICs are.
        Otherwise, the numerical wormhole solution, (r, f, u, ud, v, vd, φ, φd, χ, χd, h, flux2),
        where all but flux2 are arrays of length `nr`.

    """

    # Initial condition for f0 and the value of flux2^2 are set by u0, v0, φ0 and χ1 (f0m2 := f0**(-2))
    f0m2 = 3 - 1/3 * q0**2 * V_T11(u0, v0)
    flux2sqr = 6*np.exp(-4*u0+φ0) * q0**6 * (1 + f0m2 * (1 - (1/6)*q0**2 * np.exp(2*φ0) * χ1**2))

    # Regular solutions must have f0^2 > 0 and flux2^2 > 0
    if f0m2 < 0 or flux2sqr < 0:
        # Return values of f0m2 and flux2sqr for use in shooting method
        return [f0m2, flux2sqr]

    f0 = f0m2**(-1/2)
    flux2 = np.sqrt(flux2sqr)

    # Use series solutions (e.g. u = u0 + 1/2*udd0 * r^2 + ...)
    # to get initial conditions at 0 < rmin << q0
    udd0 = (f0**2 / 16) * ( dVdu_T11(u0, v0) -   dVdv_T11(u0, v0)) \
           + (1/8)*flux2**2 * np.exp(4*u0-φ0) * f0**2/q0**8
    vdd0 = (f0**2 / 16) * (-dVdu_T11(u0, v0) + 7*dVdv_T11(u0, v0)) \
           - (1/8)*flux2**2 * np.exp(4*u0-φ0) * f0**2/q0**8
    φdd0 = -(1/2)*flux2**2 * np.exp(4*u0-φ0) * f0**2/q0**8 - np.exp(2*φ0)*χ1**2
    fdd0 = (f0**3 * q0**2 / 12) * (6*(3+f0**(-2))/q0**4 + udd0*dVdu_T11(u0, v0) + vdd0*dVdv_T11(u0, v0))
    
    f_start = f0 + (1/2)*fdd0*rmin**2
    u_start = u0 + (1/2)*udd0*rmin**2
    v_start = v0 + (1/2)*vdd0*rmin**2
    φ_start = φ0 + (1/2)*φdd0*rmin**2

    ud_start = udd0 * rmin
    vd_start = vdd0 * rmin
    φd_start = φdd0 * rmin

    χ_start = rmin * χ1
    χd_start = χ1

    h_start = rmin * f0 / q0**4

    # Initial conditions and constants
    y0 = (f_start, u_start, ud_start, v_start, vd_start, φ_start, φd_start, χ_start, χd_start, h_start)
    args = (q0, flux2)

    soln = solve_ivp(ODEs_T11, (rmin, rmax),
                     y0=y0,
                     args=args,
                     events=(f_event),  # halt if f gets too large
                     t_eval=np.geomspace(rmin, rmax, nr),
                     rtol=10**-8,
                     method='RK45'
                    )

    # Return (r, f, u, ud, v, vd, φ, φd, χ, χd, h, flux2)
    return [soln.t, *soln.y, flux2]

def objective_T11(uv0, q0, χ1, rmax, rmin=10**-6, display=None):
    """Objective function to be minimized during T11 shooting method."""

    # Unpack ICs
    u0, v0 = uv0

    # Solve EoMs for given ICs. Can pick φ0 = 0 wlg and perform an SL(2,R) transformation
    # at the end to ensure that φ -> 0 for r -> infty.
    soln = solve_T11(q0, u0, v0, 0, χ1, rmax, rmin)


    if len(soln) == 2:
        # ICs are invalid (either f0^(-2) < 0 or flux2^2 < 0): set value to drive towards admissible ICs
        value = 10**8 + abs(min(0, soln[0])) + abs(min(0, soln[1]))
    else:
        # Unpack solution
        r, f, u, ud, v, vd, φ, φd, χ, χd, h, flux2 = soln

        if r[-1] == rmax:
            # Regular solution out to r=rmax: reward if u,v and χd are approaching zero

            # Estimate the values of u,v at r=infty using
            # the expected power-law solutions, u,v ~ r^(-6) and χd ~ r^(-5)

            # uvinf_est = (u[-1] + 0.25*v[-1]) + r[-1]*(ud[-1] + 0.25*vd[-1])/8
            uinf_est = u[-1] + r[-1]*ud[-1]/6
            vinf_est = v[-1] + r[-1]*vd[-1]/6
            # φinf_est = φ[-1] + r[-1]*φd[-1]/4
            χdinf_est = χd[-1] + r[-1]*((χd[-1] - χd[-2])/(r[-1] - r[-2]))/5

            # Reward when the extrapolated values φ(infty) and (u+0.1φ)(infty) are small
            # value -= 1/(1 + uvinf_est**2 + vinf_est**2 + χinf_dist**2)
            value = 1 - 1/(1 + uinf_est**2 + vinf_est**2 + χdinf_est**2)

        else:
            # Singular solution: penalize to drive towards nonsingular u,v,φ,...
            value = 1 + 10**3 * (rmax/r[-1] - 1) + u[-1]**2 + v[-1]**2

    # Print ICs and objective function
    if display is 'progress':
        print('{:19.15f} {:19.15f} \t {:8.4f} {:46.40f}'.format(*uv0, r[-1], value))

    return value

def wormhole_T11(q0, χ1, rmax, rmin=10**-6, nr=1000, xatol=10**-10, display=None):
    """Return optimal solution out to r=`rmax` for `q0`, `χ1` and with r=0 boundary conditions found using shooting method.
    
    Parameters
    ----------
    q0 : float
        Wormhole size in AdS units.
    χ1 : float
        Axion derivative at r=0.
    rmax : float
        End point of integration.
    rmin : float (optional)
        Start point of integration (`rmin` << `q0`).
    nr : int (optional)
        Number of r > 0 values at which to evaluate the solutions.
    xatol : float (optional)
        Option for Nelder-Mead method.
    display : [None, 'quiet', 'summary', 'progress'] (optional)
        Controls what information about the shooting method is printed.
            None (default) : nothing printed
            'quiet'        : single line giving parameters and final value of objective function
            'summary'      : summary info for final optimization, including optimal (u0, v0)
            'progress'     : (u0, v0) and objective function for every step of the optimization

    Returns
    -------
    soln : array
        The numerical wormhole solution, (r, f, u, ud, v, vd, φ, φd, χ, χd, h, flux2), where all but flux2
        are arrays of length 2*`nr`-1 (being symmetrized on the domain -`rmax` < r < `rmax`).
    
    """

    if display is not None:
        print('Optimizing (u0, v0) for (q0, χ1, rmax) = ({:.4f}, {:.4f}, {:.4f})'.format(q0, χ1, rmax), end='')

    # Perform shooting method a few times:
    #  - If q0 < 1 it helps to first integrate out to r=q0 with low precision
    #  - Integrate out to r=max(1, q0) where AdS scaling solutions begin with low precision
    #  - Finally, integrate out to r=rmax at higher precision
    rmax_list = [max(1, q0), rmax]
    xatol_list = [0.1, xatol]
    

    # if q0 < 1:
    #     rmax_list.insert(0, q0)
    #     xatol_list.insert(0, 0.1)

    # Keep track of best ICs (initial guess made with hindsight)
    u0_best = min(-0.06, +0.2446*np.log(q0) + 0.1143)
    v0_best = max(+0.12, -0.2514*np.log(q0) + 0.0969)
    uv0_best = [u0_best, v0_best]
    

    for rm, xat in zip(rmax_list, xatol_list):
        # Shooting method: optimize (u0,v0) to match AdS BCs
        opt = minimize(lambda uv0: objective_T11(uv0, q0, χ1, rm, rmin, display),
                       x0=uv0_best,
                       method='Nelder-Mead',
                       options={'maxfev': 1000, 'xatol': xat}
                      )
        # Record best (u0,v0) found
        uv0_best = opt.x

        if opt.fun > 1:
            print('\tFailed to converge')
            return None, opt.fun

    # Print results
    if display is 'quiet':
        print('\tDONE: value = {:.4g}'.format(opt.fun))

    elif display is 'summary':
        print('\n{:>14} : {}'.format('success', opt.success))
        print('{:>14} : {}'.format('f_eval', opt.nfev))
        print('{:>14} : {:+.10g}'.format('u0', uv0_best[0]))
        print('{:>14} : {:+.10g}'.format('v0', uv0_best[1]))
        print('{:>14} : {:.10g}'.format('value', opt.fun))


    # Get numerical solution for optimial initial conditions
    soln = solve_T11(q0, *uv0_best, 0, χ1, rmax, rmin, nr)
    r, f, u, ud, v, vd, φ, φd, χ, χd, h, flux2 = soln

    # Use an SL(2,R) transformation to set φ(infty) = 0
    # First estimate the current value of φ(infty)
    mask = (r > rmax/2)
    popt, pcov = curve_fit(masslessScalarFit, r[mask]/q0, φ[mask])
    φinf = popt[0]

    # Next shift φ while simultaneously rescaling χ and the flux
    soln[6] -= φinf
    soln[8] *= np.exp(φinf)
    soln[9] *= np.exp(φinf)
    soln[11] *= np.exp(-φinf/2)

    # Return symmetrized solution on -rmax < r < rmax
    return symmetrize_T11(soln), opt.fun

def symmetrize_T11(soln):
    """Extend T11 solutions from 0 < r < rmax to -rmax < r < rmax."""

    r, f, u, ud, v, vd, φ, φd, χ, χd, h, flux2 = soln
    r[0] = 0

    # Make (anti-)symmetric about r=0
    r = np.append(-r[-1:0:-1], [r])
    f = np.append( f[-1:0:-1], [f])
    u = np.append( u[-1:0:-1], [u])
    v = np.append( v[-1:0:-1], [v])
    φ = np.append( φ[-1:0:-1], [φ])
    χ = np.append(-χ[-1:0:-1], [χ])
    h = np.append(-h[-1:0:-1], [h])

    ud = np.append(-ud[-1:0:-1], [ud])
    vd = np.append(-vd[-1:0:-1], [vd])
    φd = np.append(-φd[-1:0:-1], [φd])
    χd = np.append( χd[-1:0:-1], [χd])

    return r, f, u, ud, v, vd, φ, φd, χ, χd, h, flux2

def massless_approx_T11(q0):
    
    # Taking h(0) = 0, first compute h(inf)
    h_inf, err = quad(lambda x: q0**-3 * x**3 * (q0**2 + x**2 - (1+q0**2)*x**8)**(-1/2),
                      0, 1
                     )

    # φ = -4u = 4v has profile exp(φ) = cos(sqrt(-c/2)*h) / cos(sqrt(-c/2)*hinf),
    #   so φ(0) = -log[cos(sqrt(-c/2)*hinf)]
    c_abs = 24*q0**6 * (1+q0**2)
    flux2 = np.sqrt(c_abs) / np.cos(np.sqrt(c_abs/2) * h_inf)
    φ0 = -np.log(np.cos(np.sqrt(c_abs/2) * h_inf))
    u0 = -φ0/4
    v0 = φ0/4

    return u0, v0, φ0, flux2

def frozen_approx_T11(q0):

    # Taking h(0) = 0, first compute h(inf)
    h_inf, err = quad(lambda x: q0**-3 * x**3 * (q0**2 + x**2 - (1+q0**2)*x**8)**(-1/2),
                      0, 1
                     )

    # u=v=0 and exp(φ/2) = cos(1/2 * sqrt(-c)*h) / cos(1/2 * sqrt(-c)*hinf),
    #   so φ(0) = -2*log[cos(1/2 * sqrt(-c)*hinf)]
    c_abs = 24*q0**6 * (1 + q0**2)
    flux2 = np.sqrt(c_abs) / np.cos(0.5*np.sqrt(c_abs) * h_inf)
    φ0 = -2*np.log(np.cos(0.5*np.sqrt(c_abs) * h_inf))

    return 0, 0, φ0, flux2

# def ODE_φ(r, y, q0, f3sqr):

#     φ, φd = y

#     q = Q(r, q0)
#     qd = Qd(r, q0)
#     qdd = Qdd(r, q0)

#     φ_d = φd

#     f = qd / np.sqrt(1 + q**2 - q0**6 * (1+q0**2)/q**6)

#     # Equation of motion
#     prefactor = qdd/qd - qd/q - f**2/(q*qd) * (3 + 4*q**2)
    
#     φd_d = prefactor*φd - f3sqr*np.exp(-φ) * f**2/(2*q**8)

#     return φ_d, φd_d


# def wormhole_frozen_T11(q0, rmax, nr=1000):

#     rmin = 10**-8

#     r = np.geomspace(rmin, rmax, nr)

#     q = Q(r, q0)
#     qd = Qd(r, q0)

#     c_abs = 24*q0**6 * (1 + q0**2)

#     f = qd / np.sqrt(1 + q**2 - q0**6 * (1+q0**2)/q**6)
#     f[0] = 1 / np.sqrt(3 + 4*q0**2)

#     u0, v0, φ0, f3, h_inf = frozen_approx_T11(q0)

#     f3 = np.sqrt(c_abs) / np.cos(0.5*np.sqrt(c_abs) * h_inf)


#     soln = solve_ivp(ODE_φ, (rmin, rmax),
#                      y0=[φ0, 0],
#                      args=[q0, f3**2],
#                      t_eval=r,
#                      rtol=10**-12,
#                      method='RK45'
#                     )
#     φ, φd = soln.y

#     φ -= φ[-1] + r[-1]*φd[-1]/4

#     u = 0*r
#     ud = 0*r
#     v = 0*r
#     vd = 0*r

#     soln = [r, f, u, ud, v, vd, φ, φd, 0*r, f3**2]

#     return symmetrize_T11(soln)
