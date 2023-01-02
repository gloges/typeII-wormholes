"""Construct wormhole solutions for massive type IIA on S3xS3 and for type IIB on T(1,1)."""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, curve_fit


# Conformal dimensions for light/heavy linear combinations
# of u and φ for S3xS3 which have mass-squareds 6 and 20.
Δ1_S3S3 = (3/2) + np.sqrt((3/2)**2 + 6)
Δ2_S3S3 = (3/2) + np.sqrt((3/2)**2 + 20)


def Q(r, q0):
    """Returns q=sqrt(q0^2 + r^2).""" 
    return np.sqrt(q0**2 + r**2)

def Qd(r, q0):
    """Returns derivative of q=sqrt(q0^2 + r^2)."""
    return r / Q(r, q0)

def Qdd(r, q0):
    """Returns second derivative of q=sqrt(q0^2 + r^2)."""
    return q0**2 / Q(r, q0)**3

def f_event(r, y, q0, charge):
    """Stops integration if f >> 1/q (appropriate for AdS BCs)."""
    f = y[0]
    return f - 3/Q(r, q0)
f_event.terminal = True


#SECTION - massive type IIA on S3xS3

def V_S3S3(u, φ):
    """Scalar potential V(u,φ) for S3xS3."""
    return -12*np.exp(-4*u) + 5*np.exp(-9*u-φ/2) + np.exp(-3*u+5*φ/2)

def dVdu_S3S3(u, φ):
    """u-derivative of scalar potential V(u,φ) for S3xS3."""
    return 48*np.exp(-4*u) - 45*np.exp(-9*u-φ/2) - 3*np.exp(-3*u+5*φ/2)

def dVdφ_S3S3(u, φ):
    """φ-derivative of scalar potential V(u,φ) for S3xS3."""
    return -(5/2)*np.exp(-9*u-φ/2) + (5/2)*np.exp(-3*u+5*φ/2)

def ODEs_S3S3(r, y, q0, charge):
    """ODEs describing the evolution of y=(f, u, u', φ, φ', h) for S3xS3.

    Args:
        r (float): Radial coordinate.
        y (list): Array of scalar functions of r, y=(f, u, u', φ, φ', h).
        q0 (float): Wormhole size.
        charge (float): Axion charge associated to the RR field C3.

    Returns:
        list: Derivatives of y.
    """

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
    ud_d = prefactor*ud + f**2 * ((1/12)*dVdu_S3S3(u, φ) + (1/8)*charge**2 * np.exp(3*u-φ/2)/q**6)
    φd_d = prefactor*φd + f**2 * (       dVdφ_S3S3(u, φ) - (1/4)*charge**2 * np.exp(3*u-φ/2)/q**6)
    h_d  = f/q**3
    
    # Return derivatives of f, u, ud, φ, φd and h
    return f_d, u_d, ud_d, φ_d, φd_d, h_d

def solve_S3S3(q0, u0, φ0, rmax, rmin=10**-6, nr=1000, rtol=10**-8):
    """Solves the equations of motion for S3xS3.

    After specifying the wormhole size, q0, and initial conditions
    for u and φ at r=0, the equations of motion are solved from r=rmin
    to r=rmax. Integration is halted if the geometry becomes singular.

    Args:
        q0 (float): Wormhole size.
        u0 (float): Initial condition for u at r=0.
        φ0 (float): Initial condition for φ at r=0.
        rmax (float): End point of integration.
        rmin (float, optional): Start point of integration, regularizing
            the singular equations of motion at r=0. Defaults to 10**-6.
        nr (int, optional): Number of geometrically-spaced values for r>0
            at which to evaluate the solution. Defaults to 1000.
        rtol (float, optional): Relative tolerance used for solve_ivp()
            with RK45 method. Defaults to 10**-8.

    Returns:
        array: If the initial conditions are invalid because f0**(-2) < 0,
            the array (f0**-2) for use in the shooting method. Otherwise,
            the numerical wormhole solution, (r, f, u, ud, φ, φd, h, charge),
            where all but 'charge' are arrays of length nr.
    """

    # Initial condition for f0 and the value of charge^2 are set by u0 and φ0 (f0m2 := f0**(-2))
    f0m2 = 2 - (1/2)*q0**2 * V_S3S3(u0, φ0)
    charge_sqr = 4*np.exp(-3*u0+φ0/2) * q0**4 * (1 + f0m2)
    
    # Regular solutions must have f0^2 > 0. If not, return value of f0m2 for use in shooting method
    if f0m2 < 0:
        return [f0m2]

    f0 = f0m2**(-1/2)
    charge = np.sqrt(charge_sqr)

    # Use series solutions (e.g. u = u0 + 1/2*udd0 * r^2 + ...)
    # to get initial conditions at r=rmin << q0
    udd0 = f0**2 * ( 3/(2*q0**2) - (1/4)*V_S3S3(u0, φ0) + (1/12)*dVdu_S3S3(u0, φ0))
    φdd0 = f0**2 * (-3/q0**2     + (1/2)*V_S3S3(u0, φ0) +        dVdφ_S3S3(u0, φ0))
    fdd0 = (f0**3 * q0**2 / 8) * (8/q0**4 + udd0*dVdu_S3S3(u0, φ0) + φdd0*dVdφ_S3S3(u0, φ0))

    f_start = f0 + (1/2)*fdd0 * rmin**2
    u_start = u0 + (1/2)*udd0 * rmin**2
    φ_start = φ0 + (1/2)*φdd0 * rmin**2

    ud_start = udd0 * rmin
    φd_start = φdd0 * rmin

    h_start = rmin * f0 / q0**3

    # Solve!
    soln = solve_ivp(ODEs_S3S3, (rmin, rmax),
                     y0=(f_start, u_start, ud_start, φ_start, φd_start, h_start),
                     args=(q0, charge),
                     events=(f_event),  # halt if f gets too large
                     t_eval=np.geomspace(rmin, rmax, nr),
                     rtol=rtol,
                     method='RK45'
                    )

    # Return (r, f, u, ud, φ, φd, h, charge)
    return [soln.t, *soln.y, charge]

def objective_S3S3(uφ0, q0, rmax, rmin=10**-6, rtol=10**-8, display_progress=False):
    """Objective function to be minimized during S3xS3 shooting method.

    This function quantifies how well the solution of the equations of
    motion for the given wormhole size and initial conditions matches
    to AdS boundary conditions at r -> infty. The required boundary
    conditions are f ~ 1/r, u ~ 1/r^Δ and φ ~ 1/r^Δ, while the heavy
    mode goes to zero more quickly: 10u+φ ~ 1/r^6. (The conformal dimension
    of the light mode is Δ = (3/2) + sqrt((3/2)**2 + 6) = 4.3722...)
    
    There are three cases, with a value of zero being optimal:
    
      * The initial conditions are invalid, i.e. f0**(-2) < 0. Returns
            C1 + |f0**(-2)| to drive towards admissible initial conditions.
      * The solution is singular, only reaching r=r_div < rmax. Returns
            1 + C2*(rmax/r_div - 1) + u_div**2 + φ_div**2 to drive
            towards parameters for which the solution is nonsingular
            with u and φ approaching zero.
      * The solution reaches r=rmax. Returns 1 - 1/(1 + φinf**2 + uφinf**2),
            where φinf = φ(r=infty) and uφinf = (u+0.1*φ)(r=infty) are
            estimated using the known power-law behaviors of the scalars.

    Args:
        uφ0 (list): Initial conditions for u and φ at r=0, uφ0=(u0, φ0).
        q0 (float): Wormhole size.
        rmax (float): End point of integration.
        rmin (float, optional): Start point of integration, regularizing
            the singular equations of motion at r=0. Defaults to 10**-6.
        rtol (float, optional): Relative tolerance used for solve_ivp()
            with RK45 method. Defaults to 10**-8.
        display_progress (bool, optional): To display u0, φ0, r_div
            and resulting value of the objective function. Defaults to False.

    Returns:
        float: Value of objective function.
    """

    # Solve EoMs for given ICs
    soln = solve_S3S3(q0, *uφ0, rmax, rmin, rtol=rtol)

    if len(soln) == 1:
        # ICs are invalid (i.e. f0^(-2) < 0): set value to drive towards admissible ICs
        value = 10**8 + abs(soln[0])
    else:
        # Unpack solution
        r, f, u, ud, φ, φd, h, charge = soln

        if r[-1] < rmax:
            # Singular solution: penalize to drive towards nonsingular
            value = 1 + 10**3 * (rmax/r[-1] - 1) + u[-1]**2 + φ[-1]**2
        else:
            # Regular solution out to r=rmax: reward if u,φ are approaching zero

            # Estimate the values of φ and u+0.1φ (the heavy mode) at r=infty using
            # the expected power-law solutions, φ ~ 1/r^Δ and (u+0.1φ) ~ 1/r^6.
            # For example, solving φ = φinf + A/r^Δ + (subleading) and
            # φ' = -Δ*A/r^(Δ+1) + (subleading) for φinf gives φinf ~= φ + r*φ'/Δ.
            φinf_est = φ[-1] + r[-1]*φd[-1]/Δ1_S3S3
            uφinf_est = (u[-1] + 0.1*φ[-1]) + r[-1]*(ud[-1] + 0.1*φd[-1])/6

            # Reward when the extrapolated values φ(infty) and (u+0.1φ)(infty) are small
            value = 1 - 1/(1 + φinf_est**2 + uφinf_est**2)

    if display_progress:
        # Display ICs, end point of integration and objective function
        print('{:19.15f} {:19.15f} \t {:8.4f} {:46.40f}'.format(*uφ0, r[-1], value))

    return value

def wormhole_S3S3(q0, rmax, rmin=10**-6, nr=1000, rtol=10**-8, xatol=10**-12,
                  display_summary=False, display_progress=False):
    """Finds a S3xS3 wormhole of size q0.

    Values for u and φ at r=0 are optimized using a shooting method out to rmax
    so that the solution matches on to AdS boundary conditions with u and φ going
    to zero.

    Args:
        q0 (float): Wormhole size.
        rmax (float): End point of integration.
        rmin (float, optional): Start point of integration, regularizing
            the singular equations of motion at r=0. Defaults to 10**-6.
        nr (int, optional): Number of geometrically-spaced values for r>0
            at which to evaluate the solution. Defaults to 1000.
        rtol (float, optional): Relative tolerance used for solve_ivp()
            with RK45 method. Defaults to 10**-8.
        xatol (float, optional): Option for Nelder-Mead method. Defaults to 10**-12.
        display_summary (bool, optional): Display summary information about
            shooting method. Defaults to False.
        display_progress (bool, optional): Display step-by-step information during
            shooting method. Defaults to False.

    Returns:
        array: The numerical wormhole solution, (r, f, u, ud, φ, φd, h, charge),
            where all but 'charge' are arrays of length 2*nr (having been
            symmetrized on the domain -rmax < r < rmax).
        float: Final value of the objective function.
    """

    if display_summary:
        print('S3xS3: (q0, rmax) = ({:.6f}, {:.6f})...'.format(q0, rmax))

    # Perform shooting method a few times, increasing stop point of integration with each step
    rmax_list = [max(1, q0), np.sqrt(max(1, q0)*rmax), rmax]
    xatol_list = [0.0001, np.sqrt(0.0001*xatol), xatol]

    # Keep track of best ICs (initial guess made with hindsight)
    u0_best = -0.05 - 0.73/(1 + 13.5/q0)
    φ0_best = +1.56 - 1.36/(1 + 0.12/q0)
    uφ0_best = [u0_best, φ0_best]

    for rm, xat in zip(rmax_list, xatol_list):
        # Shooting method: optimize (u0,φ0) to match AdS BCs
        opt = minimize(lambda uφ0: objective_S3S3(uφ0, q0, rm, rmin, rtol, display_progress),
                       x0=uφ0_best,
                       method='Nelder-Mead',
                       options={'maxfev': 1000, 'xatol': xat}
                      )
        # Record best (u0,φ0) found
        uφ0_best = opt.x

    # Print summary of results
    if display_summary:
        print('{:>14} : {}'.format('success', opt.success))
        print('{:>14} : {}'.format('f_eval', opt.nfev))
        print('{:>14} : {:+.10f}'.format('u0', uφ0_best[0]))
        print('{:>14} : {:+.10f}'.format('φ0', uφ0_best[1]))
        print('{:>14} : {:.10g}'.format('value', opt.fun))

    # Get numerical solution for optimial initial conditions
    soln = solve_S3S3(q0, *uφ0_best, rmax, rmin, nr, rtol)

    # Return symmetrized solution on -rmax < r < rmax
    # and final value of objective function
    return symmetrize_S3S3(soln), opt.fun

def symmetrize_S3S3(soln):
    """Extend S3xS3 solutions from 0 < r < rmax to -rmax < r < rmax."""

    r, f, u, ud, φ, φd, h, charge = soln

    # Make functions even/odd about r=0
    r = np.append(-r[-1:0:-1], [r])
    f = np.append( f[-1:0:-1], [f])
    u = np.append( u[-1:0:-1], [u])
    φ = np.append( φ[-1:0:-1], [φ])
    h = np.append(-h[-1:0:-1], [h])

    ud = np.append(-ud[-1:0:-1], [ud])
    φd = np.append(-φd[-1:0:-1], [φd])

    return r, f, u, ud, φ, φd, h, charge

def massless_approx_S3S3(q0):
    """Returns (u0,φ0) and charge for the massless "approximation" (taking u,φ to vanish at infinity)."""
    
    # Taking h(0) = 0, first compute h(inf)
    h_inf, err = quad(lambda x: q0**-2 * x**2 * (q0**2 + x**2 - (1+q0**2)*x**6)**(-1/2),
                      0, 1
                     )

    # φ = -2u has profile exp(φ) = cos(1/2 * sqrt(-c)*h) / cos(1/2 * sqrt(-c)*hinf),
    #   so φ(0) = -log[cos(1/2 * sqrt(-c)*hinf)]
    c_abs = 12*q0**4 * (1+q0**2)
    charge = np.sqrt(c_abs) / np.cos(0.5 * np.sqrt(c_abs) * h_inf)
    φ0 = -np.log(np.cos(0.5 * np.sqrt(c_abs) * h_inf))
    u0 = -0.5*φ0

    return u0, φ0, charge

def myLightMode_S3S3(r, a, b):
    """Fitting function for a rescaling of the light S3xS3 mode, (r^Δ1)*(5φ-6u), for r->infty."""
    return a + b/r**(6-Δ1_S3S3)

def fitLightMode_S3S3(r, u, φ, mask):
    """Returns fit and string for the light S3xS3 mode, 5φ-6u, for data where mask=True."""

    # Fit (r^Δ1)*(5φ-6u) to a function of the form a+b/r^(6-Δ1)
    popt, pcov = curve_fit(myLightMode_S3S3,
                           r[mask],
                           r[mask]**Δ1_S3S3 * (5*φ[mask]-6*u[mask])
                          )
    
    rr = np.geomspace(min(r[mask]), max(r[mask]), 1000)
    ff = myLightMode_S3S3(rr, *popt) / rr**Δ1_S3S3

    # Create a formatted string of the best-fit function
    a, b = popt
    label = f'$({a:.2f})$' + r'$/r^{\Delta_1}$' + f'$+({b:.2f})/r^6$'

    return rr, ff, label

def myHeavyMode_S3S3(r, a, b):
    """Fitting function for a rescaling of the heavy S3xS3 mode, (r^6)*(-φ-10u), for r->infty."""
    return a + b/r**(Δ2_S3S3-6)

def fitHeavyMode_S3S3(r, u, φ, mask):
    """Returns fit and string for the heavy S3xS3 mode, -φ-10u, for data where mask=True."""

    # Fit (r^6)*(-φ-10u) to a function of the form a+b/q^(Δ2-6)
    popt, pcov = curve_fit(myHeavyMode_S3S3,
                           r[mask],
                           r[mask]**6 * (-φ[mask]-10*u[mask])
                          )
    
    rr = np.geomspace(min(r[mask]), max(r[mask]), 1000)
    ff = myHeavyMode_S3S3(rr, *popt) / rr**6

    # Create a formatted string of the best-fit function
    a, b = popt
    label = f'$({a:.2f})/r^6+({b:.2f})$' + r'$/r^{\Delta_2}$'

    return rr, ff, label

#!SECTION

#SECTION - type IIB on T11

def V_T11(u, v):
    """Scalar potential V(u,v) for T11."""
    return 2*np.exp(-8/3*(4*u+v)) * (2*np.exp(4*u+4*v) - 12*np.exp(6*u+2*v) + 4)

def dVdu_T11(u, v):
    """u-derivative of scalar potential V(u,v) for T11."""
    return -(16/3)*np.exp(-8/3*(4*u+v)) * (5*np.exp(4*u+4*v) - 21*np.exp(6*u+2*v) + 16)

def dVdv_T11(u, v):
    """v-derivative of scalar potential V(u,v) for T11."""
    return (16/3)*np.exp(-8/3*(4*u+v)) * (np.exp(4*u+4*v) + 3*np.exp(6*u+2*v) - 4)

def ODEs_T11(r, y, q0, charge2):
    """ODEs describing the evolution of y=(f, u, u', v, v', φ, φ', χ, χ', h) for T11.

    Args:
        r (float): Radial coordinate.
        y (list): Array of scalar functions of r, y=(f, u, u', v, v', φ, φ', χ, χ', h).
        q0 (float): Wormhole size.
        charge2 (float): Axion charge associated with the RR field C2.

    Returns:
        list: Derivatives of y.
    """

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
    dVdu = dVdu_T11(u, v)
    dVdv = dVdv_T11(u, v)
    charge_term_1 = charge2**2 * np.exp(4*u+φ) * (χ**2 - np.exp(-2*φ)) / q**8
    charge_term_2 = charge2**2 * np.exp(4*u+φ) * (χ**2 + np.exp(-2*φ)) / q**8
    charge_term_3 = charge2**2 * np.exp(4*u-φ) * χ / q**8
    
    f_d  = (prefactor + 4*qd/q) * f
    ud_d = prefactor*ud + f**2 * ( (1/16)*dVdu - (1/16)*dVdv - (1/8)*charge_term_1)
    vd_d = prefactor*vd + f**2 * (-(1/16)*dVdu + (7/16)*dVdv + (1/8)*charge_term_1)
    φd_d = prefactor*φd - np.exp(2*φ)*χd**2 - (1/2)*f**2 * charge_term_2
    χd_d = prefactor*χd - 2*φd*χd + f**2 * charge_term_3

    h_d = f/q**4

    # Return derivatives of f, u, ud, v, vd, φ, φd, χ, χd and h
    return f_d, u_d, ud_d, v_d, vd_d, φ_d, φd_d, χ_d, χd_d, h_d

def solve_T11(q0, u0, v0, φ0, χ1, rmax, rmin=10**-6, nr=1000, rtol=10**-8):
    """Solves the equations of motion for T11.

    After specifying the wormhole size, q0, and initial conditions
    for u, v, φ and dχ/dr at r=0, the equations of motion are solved from
    r=rmin to r=rmax. Integration is halted if the geometry becomes singular.

    Args:
        q0 (float): Wormhole size.
        u0 (float): Initial condition for u at r=0.
        v0 (float): Initial condition for v at r=0.
        φ0 (float): Initial condition for φ at r=0.
        χ1 (float): Initial condition for dχ/dr at r=0.
        rmax (float): End point of integration.
        rmin (float, optional): Start point of integration, regularizing
            the singular equations of motion at r=0. Defaults to 10**-6.
        nr (int, optional): Number of geometrically-spaced values for r>0
            at which to evaluate the solution. Defaults to 1000.
        rtol (float, optional): Relative tolerance used for solve_ivp()
            with RK45 method. Defaults to 10**-8.

    Returns:
        array: If the initial conditions are invalid (either f0**-2 < 0
            or charge2**2 < 0), the array (f0**-2, charge2**2) for use in the
            shooting method. Otherwise, the numerical wormhole solution,
            (r, f, u, ud, v, vd, φ, φd, χ, χd, h, charge), where all but
            'charge2' are arrays of length nr.
    """

    # Initial condition for f0 and the value of charge2^2 are set
    # by u0, v0, φ0 and χ1 (f0m2 := f0**(-2))
    f0m2 = 3 - 1/3 * q0**2 * V_T11(u0, v0)
    charge2_sqr = 6*np.exp(-4*u0+φ0) * q0**6 \
                  * (1 + f0m2 * (1 - (1/6)*q0**2 * np.exp(2*φ0) * χ1**2))

    # Regular solutions must have f0^2 > 0 and charge2^2 > 0
    if f0m2 < 0 or charge2_sqr < 0:
        # Return values of f0m2 and charge2_sqr for use in shooting method
        return [f0m2, charge2_sqr]

    f0 = f0m2**(-1/2)
    charge2 = np.sqrt(charge2_sqr)

    # Use series solutions (e.g. u = u0 + 1/2*udd0 * r^2 + ...)
    # to get initial conditions at r=rmin << q0
    dVdu = dVdu_T11(u0, v0)
    dVdv = dVdv_T11(u0, v0)
    charge_term_1 = charge2**2 * np.exp(4*u0-φ0) / q0**8

    udd0 = f0**2 * ( (1/16)*dVdu - (1/16)*dVdv + (1/8)*charge_term_1)
    vdd0 = f0**2 * (-(1/16)*dVdu + (7/16)*dVdv - (1/8)*charge_term_1)
    φdd0 = -np.exp(2*φ0)*χ1**2 - (1/2)*f0**2 * charge_term_1

    fdd0 = (f0**3 * q0**2 / 12) * (6*(3+f0**(-2))/q0**4 + udd0*dVdu + vdd0*dVdv)
    
    f_start = f0 + (1/2)*fdd0*rmin**2
    u_start = u0 + (1/2)*udd0*rmin**2
    v_start = v0 + (1/2)*vdd0*rmin**2
    φ_start = φ0 + (1/2)*φdd0*rmin**2

    ud_start = udd0 * rmin
    vd_start = vdd0 * rmin
    φd_start = φdd0 * rmin

    # χ = χ1*r + ...
    χ_start = rmin * χ1
    χd_start = χ1

    h_start = rmin * f0 / q0**4

    # Initial conditions    
    y0 = (f_start, u_start, ud_start, v_start, vd_start,
          φ_start, φd_start, χ_start, χd_start, h_start)

    # Solve!
    soln = solve_ivp(ODEs_T11, (rmin, rmax),
                     y0=y0,
                     args=(q0, charge2),
                     events=(f_event),  # halt if f gets too large
                     t_eval=np.geomspace(rmin, rmax, nr),
                     rtol=rtol,
                     method='RK45'
                    )

    # Return (r, f, u, ud, v, vd, φ, φd, χ, χd, h, charge2)
    return [soln.t, *soln.y, charge2]

def objective_T11(uv0, q0, χ1, rmax, rmin=10**-6, rtol=10**-8, display_progress=False):
    """Objective function to be minimized during T11 shooting method.

    This function quantifies how well the solution of the equations of
    motion for the given wormhole size and initial conditions matches
    to AdS boundary conditions at r -> infty. The required boundary
    conditions are f ~ 1/r, u ~ 1/r^6, v ~ 1/r^6, φ ~ 1/r^4 and χ ~ 1/r^4,
    while the heavy mode 4u+v goes to zero more quickly: 4u+v ~ log(r)/r^8.
    Without loss of generality φ(r=0) is chosen to be zero.
    
    There are three cases, with a value of zero being optimal:
    
      * The initial conditions are invalid, i.e. f0**(-2) < 0 or charge2**2 < 0.
            Returns C1 + |min(0, f0**(-2))| + |min(0, charge2**2)| to drive
            towards admissible initial conditions.
      * The solution is singular, only reaching r=r_div < rmax. Returns
            1 + C2*(rmax/r_div - 1) + u_div**2 + v_div**2 to drive
            towards parameters for which the solution is nonsingular
            with u and v approaching zero.
      * The solution reaches r=rmax. Returns 1 - 1/(1 + uvinf**2 + uinf**2),
            where uvinf = (u+0.25v)(r=infty) and vinf = v(r=infty) are
            estimated using the known power-law behaviors of the scalars.
            Checking these two conditions is (in practice) enough to ensure
            the other boundary conditions are satisfied as well.

    Args:
        uv0 (list): Initial conditions for u and v at r=0, uv0=(u0,v0).
        q0 (float): Wormhole size.
        χ1 (float): Initial condition for dχ/dr at r=0.
        rmax (float): End point of integration.
        rmin (float, optional): Start point of integration, regularizing
            the singular equations of motion at r=0. Defaults to 10**-6.
        rtol (float, optional): Relative tolerance used for solve_ivp()
            with RK45 method. Defaults to 10**-8.
        display_progress (bool, optional): To display u0, v0, r_div
            and resulting value of the objective function. Defaults to False.

    Returns:
        float: Value of objective function.
    """    

    # Unpack initial conditions
    u0, v0 = uv0

    # Solve EoMs for given ICs. Can pick φ0 = 0 wlg and perform
    # an SL(2,R) transformation to ensure φ(infty) = 0 afterwards.
    soln = solve_T11(q0, u0, v0, 0, χ1, rmax, rmin, rtol=rtol)

    if len(soln) == 2:
        # ICs are invalid (i.e. f0^(-2) < 0): set value to drive towards admissible ICs
        value = 10**8 + abs(min(0, soln[0])) + abs(min(0, soln[1]))
    else:
        # Unpack solution
        r, f, u, ud, v, vd, φ, φd, χ, χd, h, charge2 = soln

        if r[-1] < rmax:
            # Singular solution: penalize to drive towards nonsingular
            value = 1 + 10**3 * (rmax/r[-1] - 1) + u[-1]**2 + v[-1]**2

        else:
            # Regular solution out to r=rmax: reward if u and v are approaching zero

            # Estimate the values of u,v at r=infty using the expected power-law solutions,
            # u,v ~ 1/r^6 and u+0.25v ~ log(r)/r^8 (ignore log factor for simplicity)

            uvinf_est = (u[-1] + 0.25*v[-1]) + r[-1]*(ud[-1] + 0.25*vd[-1])/8
            vinf_est = v[-1] + r[-1]*vd[-1]/6

            # Reward when the extrapolated values (u+0.25v)(infty) and v(infty) are small
            value = 1 - 1/(1 + uvinf_est**2 + vinf_est**2)

    # Print ICs and objective function
    if display_progress:
        print('{:19.15f} {:19.15f} \t {:8.4f} {:46.40f}'.format(*uv0, r[-1], value))

    return value

def wormhole_T11(q0, χ1, rmax, rmin=10**-6, nr=1000, rtol=10**-8, xatol=10**-12,
                 display_summary=False, display_progress=False):
    """Finds a T11 wormhole of size q0.

    Values for u and v at r=0 are optimized using a shooting method out to rmax
    so that the solution matches on to AdS boundary conditions with u and v going
    to zero. Without loss of generality this optimization is done with the choice
    φ(r=0) = 0: an SL(2,R) transformation is performed at the end to ensure that φ
    also goes to zero at infinity, in the process rescaling χ, dχ/dr and the charge.

    Args:
        q0 (float): Wormhole size.
        χ1 (float): Initial condition for exp(φ)*(dχ/dr) at r=0.
        rmax (float): End point of integration.
        rmin (float, optional): Start point of integration, regularizing
            the singular equations of motion at r=0. Defaults to 10**-6.
        nr (int, optional): Number of geometrically-spaced values for r>0
            at which to evaluate the solution. Defaults to 1000.
        rtol (float, optional): Relative tolerance used for solve_ivp()
            with RK45 method. Defaults to 10**-8.
        xatol (float, optional): Option for Nelder-Mead method. Defaults to 10**-12.
        display_summary (bool, optional): Display summary information about
            shooting method. Defaults to False.
        display_progress (bool, optional): Display step-by-step information during
            shooting method. Defaults to False.

    Returns:
        array: The numerical wormhole solution, (r, f, u, ud, v, vd, φ, φd,
            χ, χd, h, charge), where all but 'charge2' are arrays of length 2*nr
            (having been symmetrized on the domain -rmax < r < rmax).
        float: Final value of the objective function.
    """

    if display_summary:
        print('T11: (q0, χ1, rmax) = ({:.6f}, {:.6f}, {:.6f})...'.format(q0, χ1, rmax))

    # Perform shooting method a few times, increasing the stop point of integration
    # and precision with each step.
    rmax_list = [max(1, q0), np.sqrt(max(1, q0)*rmax), rmax]
    xatol_list = [0.0001, np.sqrt(0.0001*xatol), xatol]
    
    # Keep track of best ICs (initial guess made with hindsight)
    u0_best = min(-0.06, +0.25*np.log(q0) + 0.1)
    v0_best = max(+0.12, -0.25*np.log(q0) + 0.1)
    uv0_best = [u0_best, v0_best]
    
    for rm, xat in zip(rmax_list, xatol_list):
        # Shooting method: optimize (u0,v0) to match AdS BCs
        opt = minimize(lambda uv0: objective_T11(uv0, q0, χ1, rm, rmin, rtol, display_progress),
                       x0=uv0_best,
                       method='Nelder-Mead',
                       options={'maxfev': 1000, 'xatol': xat}
                      )
        # Record best (u0,v0) found
        uv0_best = opt.x

    if opt.fun > 1:
        print('\tFailed to converge')
        return None, opt.fun

    # Print summary of results
    if display_summary:
        print('{:>14} : {}'.format('success', opt.success))
        print('{:>14} : {}'.format('f_eval', opt.nfev))
        print('{:>14} : {:+.10f}'.format('u0', uv0_best[0]))
        print('{:>14} : {:+.10f}'.format('v0', uv0_best[1]))
        print('{:>14} : {:.10g}'.format('value', opt.fun))

    # Get numerical solution for optimial initial conditions
    soln = solve_T11(q0, *uv0_best, 0, χ1, rmax, rmin, nr, rtol)
    r, f, u, ud, v, vd, φ, φd, χ, χd, h, charge2 = soln

    # Use an SL(2,R) transformation to set φ(infty) = 0
    # First estimate the current value of φ(infty) by fitting
    # the tail of the profile to a function A + B/r^4 + C/r^6
    mask = (r > rmax/1.5)
    popt, pcov = curve_fit(myMassless_T11, r[mask]/q0, φ[mask])
    φinf = popt[0]

    # Next perform the transformation
    soln[6]  -= φinf                # Shift φ
    soln[8]  *= np.exp(φinf)        # Rescale χ
    soln[9]  *= np.exp(φinf)        # Rescale χd
    soln[11] *= np.exp(-φinf/2)     # Rescale the axion charge

    # Return symmetrized solution on -rmax < r < rmax
    # and final value of objective function
    return symmetrize_T11(soln), opt.fun

def symmetrize_T11(soln):
    """Extend T11 solutions from 0 < r < rmax to -rmax < r < rmax."""

    r, f, u, ud, v, vd, φ, φd, χ, χd, h, charge2 = soln

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

    return r, f, u, ud, v, vd, φ, φd, χ, χd, h, charge2

def massless_approx_T11(q0):
    """Returns (u0,v0,φ0) and charge for the massless "approximation" (taking u,v,φ to vanish at infinity)."""
    
    # Taking h(0) = 0, first compute h(inf)
    h_inf, err = quad(lambda x: q0**-3 * x**3 * (q0**2 + x**2 - (1+q0**2)*x**8)**(-1/2),
                      0, 1
                     )

    # φ = -4u = 4v has profile exp(φ) = cos(sqrt(-c/2)*h) / cos(sqrt(-c/2)*hinf),
    #   so φ(0) = -log[cos(sqrt(-c/2)*hinf)]
    c_abs = 24*q0**6 * (1+q0**2)
    charge2 = np.sqrt(c_abs) / np.cos(np.sqrt(c_abs/2) * h_inf)
    φ0 = -np.log(np.cos(np.sqrt(c_abs/2) * h_inf))
    u0 = -φ0/4
    v0 = φ0/4

    return u0, v0, φ0, charge2

def frozen_approx_T11(q0):
    """Returns (u0,v0,φ0) and charge for the "frozen approximation" (taking φ to vanish at infinity)."""

    # Taking h(0) = 0, first compute h(inf)
    h_inf, err = quad(lambda x: q0**-3 * x**3 * (q0**2 + x**2 - (1+q0**2)*x**8)**(-1/2),
                      0, 1
                     )

    # u=v=0 and exp(φ/2) = cos(1/2 * sqrt(-c)*h) / cos(1/2 * sqrt(-c)*hinf),
    #   so φ(0) = -2*log[cos(1/2 * sqrt(-c)*hinf)]
    c_abs = 24*q0**6 * (1 + q0**2)
    charge2 = np.sqrt(c_abs) / np.cos(0.5*np.sqrt(c_abs) * h_inf)
    φ0 = -2*np.log(np.cos(0.5*np.sqrt(c_abs) * h_inf))

    return 0, 0, φ0, charge2

def myMassless_T11(r, ψinf, ψ4, ψ6):
    """Functional form for any massless scalar ψ on T11, for r->infty."""
    return ψinf + ψ4/r**4 + ψ6/r**6

def myMasslessDeriv_T11(r, ψ4, ψ6):
    """Functional form of r^5(dψ/dr) if ψ ~ 0 + ψ4/r^4 + ψ6/r^6
    is any massless scalar on T11 which goes to zero for r->infty."""
    return -4*ψ4 - 6*ψ6/r**2

def myLightMode_T11(r, a, b):
    """Fitting function for a rescaling of the light T11 mode, (r^6)*(-u+v), for r->infty."""
    return a + b/r**2

def fitLightMode_T11(r, u, v, mask):
    """Returns fit and string for the light T11 mode, -u+v, for data where mask=True."""

    # Fit (r^6)*(-u+v) to a function of the form a+b/r^2
    popt, pcov = curve_fit(myLightMode_T11,
                           r[mask],
                           r[mask]**6 * (-u[mask]+v[mask])
                          )
    
    rr = np.geomspace(min(r[mask]), max(r[mask]), 1000)
    ff = myLightMode_T11(rr, *popt) / rr**6

    # Create a formatted string of the best-fit function
    a, b = popt
    label = f'$({a:.2f})/r^6+({b:.2f})/r^8$'

    return rr, ff, label

def myHeavyMode_T11(r, a, b):
    """Fitting function for a rescaling of the heavy T11 mode, (r^8)*(-4u-v), for r->infty."""
    return a*np.log(r/b)

def fitHeavyMode_T11(r, u, v, mask):
    """Returns fit and string for the heavy T11 mode, -4u-v, for data where mask=True."""

    # Fit (r^8)*(-4u-v) to a function of the form a*log(r/b)
    popt, pcov = curve_fit(myHeavyMode_T11,
                           r[mask],
                           r[mask]**8 * (-4*u[mask]-v[mask])
                          )
    
    rr = np.geomspace(min(r[mask]), max(r[mask]), 1000)
    ff = myHeavyMode_T11(rr, *popt) / rr**8

    # Create a formatted string of the best-fit function
    a, b = popt
    label = f'$({a:.2f})\log(r/{b:.2f})/r^8$'

    return rr, ff, label

def fitMassless_T11(r, ψ, mask):
    """Returns fit and string for a massless scalar on T11 which goes to zero for r->infty, for data where mask=True."""

    # Fit (r^4)*ψ to a function of the form a+b/r^2
    popt, pcov = curve_fit(myLightMode_T11,
                           r[mask],
                           r[mask]**4 * ψ[mask]
                          )
    
    rr = np.geomspace(min(r[mask]), max(r[mask]), 1000)
    ff = myLightMode_T11(rr, *popt) / rr**4

    # Create a formatted string of the best-fit function
    a, b = popt
    label = f'$({a:.2f})/r^4+({b:.2f})/r^6$'

    return rr, ff, label

def ricci5D_T11(q0, soln):
    """Returns the 5D Ricci scalar for a T11 wormhole of size q0."""

    # Unpack and get q(r) and q'(r)
    r, f, u, ud, v, vd, φ, φd, χ, χd, h, charge2 = soln
    q  =  Q(r, q0)
    qd = Qd(r, q0)

    # Get 5D Ricci scalar as a function of r
    # (this has been simplified using the equations of motion)
    R5 = (12/q**2) * (qd**2/f**2 - 1) + 8/3 * V_T11(u, v)

    return R5

def ricci10D_T11(q0, soln):
    """Returns the 10D Ricci scalar (times l^2) for the uplift of a T11 wormhole of size q0."""

    # Unpack and get q(r)
    r, f, u, ud, v, vd, φ, φd, χ, χd, h, charge2 = soln
    q = Q(r, q0)

    # Get 10D Ricci scalar as a function of r
    # (this has been simplified using the equations of motion)
    R10 = (1/4)*np.exp(2/3*(4*u+v)) * (2*(φd**2 - np.exp(2*φ)*χd**2)/f**2 \
                                       + charge2**2 * np.exp(4*u+φ) * (χ**2 - np.exp(-2*φ)) / q**8)

    return R10

#!SECTION