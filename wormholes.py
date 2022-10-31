import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from itertools import product
from tqdm import tqdm



def Q(r, q0):
    """Returns q=sqrt(q0^2 + r^2)."""
    return np.sqrt(q0**2 + r**2)

def Qd(r, q0):
    """Returns derivative of q=sqrt(q0^2 + r^2)."""
    return r / Q(r, q0)

def Qdd(r, q0):
    """Returns second derivative of q=sqrt(q0^2 + r^2)."""
    return q0**2 / Q(r, q0)**3

def wevent(r, y, q0, f3sqr):
    """Stops integration if w -> 0."""
    w = y[0]
    return w - 0.1
wevent.terminal = True


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
    """ODEs for w, u, φ and h."""

    # Unpack
    w, u, ud, φ, φd, h = y

    q = Q(r, q0)
    qd = Qd(r, q0)
    qdd = Qdd(r, q0)
    
    # By definition
    u_d = ud
    φ_d = φd
    
    # Equations of motion
    prefactor = qdd/qd - qd/q - 1/(2*q*qd*w) * (4/q**2 - V_S3S3(u, φ))
    
    w_d  = -2*w*(prefactor + 4*qd/q)
    ud_d = prefactor*ud + 1/(q**2 * w) * ((1/12)*dVdu_S3S3(u, φ) + f4sqr*np.exp(3*u-φ/2)/(8*q**6))
    φd_d = prefactor*φd + 1/(q**2 * w) * (       dVdφ_S3S3(u, φ) - f4sqr*np.exp(3*u-φ/2)/(4*q**6))
    h_d  = 1/(q**4 * np.sqrt(abs(w)))
    
    # Return derivatives of w, u, ud, φ, φd and h
    return w_d, u_d, ud_d, φ_d, φd_d, h_d

def solve_S3S3(q0, u0, φ0, rmax, rmin=10**-16, nr=1000):
    """Solves ODEs for S3xS3 out to r=rmax for given q0 and initial conditions u0,φ0."""

    # Initial condition for w0 and the value of f4^2 are set by u0 and φ0
    w0 = 2/q0**2 - V_S3S3(u0, φ0)/2
    f4sqr = 4*np.exp(-3*u0+φ0/2) * q0**4 * (1 + q0**2 * w0)
    
    if w0 < 0:
        # Regular solutions must have w0>0
        return [w0]

    # Use series solutions (e.g. u = u0 + 1/2*udd0 * r^2 + ...)
    # to get initial conditions at 0 < rmin << q0
    udd0 = 1/(q0**2 * w0) * ( 3/(2*q0**2) - (1/4)*V_S3S3(u0, φ0) + (1/12)*dVdu_S3S3(u0, φ0))
    φdd0 = 1/(q0**2 * w0) * (-3/q0**2     + (1/2)*V_S3S3(u0, φ0) +        dVdφ_S3S3(u0, φ0))
    wdd0 = -6/q0**4 + V_S3S3(u0, φ0)/q0**2 - (1/4)*udd0*dVdu_S3S3(u0, φ0) - (1/4)*φdd0*dVdφ_S3S3(u0, φ0)
    
    w_start = w0 + (1/2)*wdd0*rmin**2

    u_start = u0 + (1/2)*udd0*rmin**2
    ud_start = udd0 * rmin

    φ_start = φ0 + (1/2)*φdd0*rmin**2
    φd_start = φdd0 * rmin

    h_start = rmin / (q0**4 * np.sqrt(w0))

    # ICs and constants
    y0 = (w_start, u_start, ud_start, φ_start, φd_start, h_start)
    args = (q0, f4sqr)

    soln = solve_ivp(ODEs_S3S3, (rmin, rmax),
                     y0=y0,
                     args=args,
                     events=(wevent),
                     t_eval=np.linspace(rmin, rmax, nr),
                     rtol=10**-16,
                     method='RK45'
                    )

    # Return (r, w, u, ud, φ, φd, h, f4sqr)
    return [soln.t, *soln.y, f4sqr]

def objective_S3S3(uφ0, q0, rmax, rmin=10**-16, nr=1000):
    """Objective function to be minimized during shooting method."""

    soln = solve_S3S3(q0, *uφ0, rmax, rmin)

    if len(soln) == 1:
        # Invlid (w0<0): return (const.)+|w0| to drive towards w0>0
        return 10**8 + abs(soln[0])
    else:
        # Unpack solution
        r, w, u, ud, φ, φd, h, f4sqr = soln
        
        Δ1 = (3/2) + np.sqrt((3/2)**2 + 6)      # Conformal dimension for light mode
        

        value = 1 + 10**3 * (rmax/r[-1] - 1)

        if r[-1] == rmax:
            value -= 1/(1 + ((u[-1] + 0.1*φ[-1]) + r[-1]*(ud[-1] + 0.1*φd[-1])/6)**2 \
                        + (φ[-1] + r[-1]*φd[-1]/Δ1)**2)
            # print('{:24.20f} {:24.20f}\t   good!: {:4.2f}\t{:46.40f}'.format(*uφ0, r[-1], value))
        else:
            value += u[-1]**2 + φ[-1]**2
            # print('{:24.20f} {:24.20f}\tsingular: {:4.2f}\t{:46.40f}'.format(*uφ0, r[-1], value))

        return value

def paramScan_S3S3(q0, u0_list, φ0_list, rmax):
    """Returns objective function for a grid of u0,φ0 values."""

    data = []

    for u0, φ0 in tqdm(product(u0_list, φ0_list), total=len(u0_list)*len(φ0_list)):
        quality = objective_S3S3([u0, φ0], q0, rmax)
        data.append([u0, φ0, quality])
    
    data = np.asarray(data).T

    return data

def wormhole_S3S3(q0, rmax_list, uφ0=[0,0], xatol=10**-16, nr=1000):
    """Return optimal solution out to r=rmax for q0 and with r=0 boundary conditions found using shooting method."""

    uφ0_best = uφ0

    for rmax in rmax_list:
        # Shooting method: determine u0,φ0 to match boundary conditions/scaling solutions at r>>q0
        opt = minimize(objective_S3S3,
                    args=(q0, rmax),
                    x0=uφ0_best,
                    method='Nelder-Mead',
                    options={'maxfev': 1000,
                             'xatol': xatol,
                            }
                    )
        uφ0_best = opt.x

        # Print results
        print('(q0,rmax,fev) = ({:6.4f},{:4.0f},{:4d})'.format(q0, rmax, opt.nfev), end=4*' ')
        print('(u0,φ0) = ({:+.16f}, {:+.16f})'.format(*uφ0_best), end=4*' ')
        print('val = {:.16}'.format(opt.fun))

    # Get numerical solution for optimial initial conditions
    soln = solve_S3S3(q0, *uφ0_best, rmax, nr=nr)
    r, w, u, ud, φ, φd, h, f4sqr = soln

    # Print results
    print('(q0,rmax,fev) = ({:6.4f},{:6.4f},{:4d})'.format(q0, rmax, opt.nfev), end=4*' ')
    print('(u0,φ0,uf,φf) = ({:+.16f}, {:+.16f}, {:+.16f}, {:+.16f})\n' \
        .format(*uφ0_best, soln[2][-1], soln[4][-1]))

    return symmetrize_S3S3(soln)

def symmetrize_S3S3(soln):
    """Extend solutions from 0 < r < rmax to -rmax < r < rmax."""

    r, w, u, ud, φ, φd, h, f4sqr = soln
    r[0] = 0

    # Make functions even/odd about r=0
    r = np.append(-r[-1:0:-1], [r])
    w = np.append( w[-1:0:-1], [w])
    u = np.append( u[-1:0:-1], [u])
    φ = np.append( φ[-1:0:-1], [φ])
    h = np.append(-h[-1:0:-1], [h])

    ud = np.append(-ud[-1:0:-1], [ud])
    φd = np.append(-φd[-1:0:-1], [φd])

    return r, w, u, ud, φ, φd, h, f4sqr


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
    """ODEs for w, u, v, φ and h."""

    # Unpack
    w, u, ud, v, vd, φ, φd, h = y

    q = Q(r, q0)
    qd = Qd(r, q0)
    qdd = Qdd(r, q0)

    # By definition
    u_d = ud
    v_d = vd
    φ_d = φd

    # Equations of motion
    prefactor = qdd/qd - qd/q - 1/(3*q*qd*w) * (9/q**2 - V_T11(u, v))
    
    w_d  = -2*w*(prefactor + 5*qd/q)
    ud_d = prefactor*ud + 1/(16*q**2 * w) * ( dVdu_T11(u, v) -   dVdv_T11(u, v) + 2*f3sqr*np.exp(4*u-φ)/q**8)
    vd_d = prefactor*vd + 1/(16*q**2 * w) * (-dVdu_T11(u, v) + 7*dVdv_T11(u, v) - 2*f3sqr*np.exp(4*u-φ)/q**8)
    φd_d = prefactor*φd - f3sqr*np.exp(4*u-φ)/(2*q**10 * w)

    h_d = 1/(q**5 * np.sqrt(abs(w)))

    return w_d, u_d, ud_d, v_d, vd_d, φ_d, φd_d, h_d

def solve_T11(q0, u0, v0, φ0, rmax, rmin=10**-4, nr=1000):
    """Solves ODEs for T11 out to r=rmax for given q0 and initial conditions u0,v0,φ0."""

    # Initial condition for w0 and the value of f3^2 are set by u0, v0 and φ0
    w0 = 3/q0**2 - V_T11(u0, v0)/3
    f3sqr = 6*np.exp(-4*u0+φ0) * q0**6 * (1 + q0**2 * w0)
    
    if w0 < 0:
        # Regular solutions must have w0>0
        return [w0]

    # Use series solutions (e.g. u = u0 + 1/2*udd0 * r^2 + ...)
    # to get initial conditions at 0 < rmin << q0
    udd0 = 1/(16*q0**2 * w0) * ( dVdu_T11(u0, v0) -   dVdv_T11(u0, v0) + 2*f3sqr*np.exp(4*u0-φ0)/q0**8)
    vdd0 = 1/(16*q0**2 * w0) * (-dVdu_T11(u0, v0) + 7*dVdv_T11(u0, v0) - 2*f3sqr*np.exp(4*u0-φ0)/q0**8)
    φdd0 = -f3sqr*np.exp(4*u0-φ0)/(2*q0**10 * w0)
    wdd0 = -12/q0**4 + V_T11(u0, v0)/q0**2 - (1/6)*udd0*dVdu_T11(u0, v0) - (1/6)*vdd0*dVdv_T11(u0, v0)
    
    w_start = w0 + (1/2)*wdd0*rmin**2

    u_start = u0 + (1/2)*udd0*rmin**2
    ud_start = udd0 * rmin
    
    v_start = v0 + (1/2)*vdd0*rmin**2
    vd_start = vdd0 * rmin

    φ_start = φ0 + (1/2)*φdd0*rmin**2
    φd_start = φdd0 * rmin

    h_start = rmin / (q0**5 * np.sqrt(w0))

    # ICs and constants
    y0 = (w_start, u_start, ud_start, v_start, vd_start, φ_start, φd_start, h_start)
    args = (q0, f3sqr)

    soln = solve_ivp(ODEs_T11, (rmin, rmax),
                     y0=y0,
                     args=args,
                     events=(wevent),
                     t_eval=np.linspace(rmin, rmax, nr),
                     rtol=10**-8,
                     method='RK45'
                    )

    # Return (r, w, u, ud, v, vd, φ, φd, h, f4sqr)
    return [soln.t, *soln.y, f3sqr]

def objective_T11(uv0, φ0, q0, rmax, rmin=10**-4, nr=1000):
    """Objective function to be minimized during shooting method."""
    
    soln = solve_T11(q0, *uv0, φ0, rmax, rmin)

    if len(soln) == 1:
        # Invlid (w0<0): return (const.)+|w0| to drive towards w0>0
        # print('{:24.20f} {:24.20f}\tinvalid: {}'.format(*uv0, soln[0]))
        return 10**16 + abs(soln[0])
    else:
        # Unpack solution
        r, w, u, ud, v, vd, φ, φd, h, f3sqr = soln

        Δ1 = (4/2) + np.sqrt((4/2)**2 + 12)     # Conformal dimension for light mode

        value = 1 + 10**3 * (rmax/r[-1] - 1)

        if r[-1] == rmax:
            value -= 1/(1 + ((u[-1] + 0.25*v[-1]) + r[-1]*(ud[-1] + 0.25*vd[-1])/8)**2 \
                        + (v[-1] + r[-1]*vd[-1]/Δ1)**2)
            # value -= 1/(1 + (u[-1]**2 + (v[-1] + r[-1]*vd[-1]/Δ1)**2))
            # value -= 1/(1 + ((u[-1] + 0.25*v[-1])**2 + (v[-1] + r[-1]*vd[-1]/Δ1)**2))
            # print('{:24.20f} {:24.20f}\t   good!: {:4.2f}\t{:46.40f}'.format(*uv0, r[-1], value))
        else:
            value += u[-1]**2 + v[-1]**2
            # print('{:24.20f} {:24.20f}\tsingular: {:4.2f}\t{:46.40f}'.format(*uv0, r[-1], value))

        return value

def paramScan_T11(q0, u0_list, v0_list, rmax):
    """Returns objective function for a grid of u0,v0 values."""

    data = []

    for u0, v0 in tqdm(product(u0_list, v0_list), total=len(u0_list)*len(v0_list)):
        quality = objective_T11([u0, v0], 0, q0, rmax)
        data.append([u0, v0, quality])
    
    data = np.asarray(data).T

    return data

def wormhole_T11(q0, rmax_list, uv0=[0,0], xatol=10**-16, nr=1000):
    """Return optimal solution out to r=rmax for q0 and with r=0 boundary conditions found using shooting method."""

    uv0_best = uv0

    for rmax in rmax_list:
        # Shooting method: determine u0,v0 to match boundary conditions/scaling solutions at r>>q0
        # Set φ0 = 0 for optimizing u0,v0 since can always absorb constant into f3^2 at the end
        opt = minimize(objective_T11,
                    args=(0, q0, rmax),
                    x0=uv0_best,
                    method='Nelder-Mead',
                    options={'maxfev': 1000,
                             'xatol': xatol,
                            }
                    )
        uv0_best = opt.x

        # Print results
        print('(q0,rmax,fev) = ({:6.4f},{:4.0f},{:4d})'.format(q0, rmax, opt.nfev), end=4*' ')
        print('(u0,v0) = ({:+.16f}, {:+.16f})'.format(*uv0_best), end=4*' ')
        print('val = {:.16}'.format(opt.fun))

    # Get numerical solution for optimial initial conditions
    soln = solve_T11(q0, *uv0_best, 0, rmax, nr=nr)
    r, w, u, ud, v, vd, φ, φd, h, f3sqr = soln

    φinf = φ[-1] + r[-1]*φd[-1]/4
    soln[6] -= φinf
    soln[9] *= np.exp(-φinf/2)

    # Print results
    print('(q0,rmax,fev) = ({:6.4f},{:4.0f},{:4d})'.format(q0, rmax, opt.nfev), end=4*' ')
    print('(u0,v0,uf,vf) = ({:+.16f}, {:+.16f}, {:+.16f}, {:+.16f})\n' \
        .format(*uv0_best, soln[2][-1], soln[4][-1]))

    return symmetrize_T11(soln)

def symmetrize_T11(soln):
    """Extend solutions from 0 < r < rmax to -rmax < r < rmax."""

    r, w, u, ud, v, vd, φ, φd, h, f3sqr = soln
    r[0] = 0

    # Make (anti-)symmetric about r=0
    r = np.append(-r[-1:0:-1], [r])
    w = np.append( w[-1:0:-1], [w])
    u = np.append( u[-1:0:-1], [u])
    v = np.append( v[-1:0:-1], [v])
    φ = np.append( φ[-1:0:-1], [φ])
    h = np.append(-h[-1:0:-1], [h])

    ud = np.append(-ud[-1:0:-1], [ud])
    vd = np.append(-vd[-1:0:-1], [vd])
    φd = np.append(-φd[-1:0:-1], [φd])

    return r, w, u, ud, v, vd, φ, φd, h, f3sqr
