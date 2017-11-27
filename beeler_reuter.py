import numpy as np

SAMPLES = 2048
C_m     = 1.0
diffCoef   = 0.001
minVlt     = -90    # mV
maxVlt     = 30     # mV

inf_names = ['xi_inf', 'm_inf', 'h_inf', 'j_inf', 'd_inf', 'f_inf']
tau_names = ['xi_tau', 'm_tau', 'h_tau', 'j_tau', 'd_tau', 'f_tau']

ab_coef = np.array([[0.0005, 0.083,  50.,    0.0,    0.0,    0.057,  1.0],   # ca_x1
           [0.0013, -0.06,  20.,    0.0,    0.0,    -0.04,  1.0],   # cb_x1
           [0.0000, 0.0,    47.,    -1.0,   47.,    -0.1,   -1.0],  # ca_m
           [40.,    -0.056, 72.,    0.0,    0.0,    0.0,    0.0],   # cb_m
           [0.126,  -.25,   77.,    0.0,    0.0,    0.0,    0.0],   # ca_h
           [1.7,    0.0,    22.5,   0.0,    0.0,    -0.082, 1.0],   # cb_h
           [0.055,  -.25,   78.0,   0.0,    0.0,    -0.2,   1.0],   # ca_j
           [0.3,    0.0,    32.,    0.0,    0.0,    -0.1,   1.0],   # cb_j
           [2*0.095,  -0.01,  -5.,    0.0,    0.0,    -0.072, 1.0],   # ca_d
           [2*0.07,   -0.017, 44.,    0.0,    0.0,    0.05,   1.0],   # cb_d
           [2*0.012,  -0.008, 28.,    0.0,    0.0,    0.15,   1.0],   # ca_f
           [2*0.0065, -0.02,  30.,    0.0,    0.0,    -0.2,   1.0]])  # cb_f

def calc_alpha_beta(v):
        v = np.outer(v, np.ones(ab_coef.shape[0]))
        return ((ab_coef[:,0] * np.exp(ab_coef[:,1] * (v+ab_coef[:,2])) +
                ab_coef[:,3] * (v+ab_coef[:,4])) /
                (np.exp(ab_coef[:,5] * (v+ab_coef[:,2])) + ab_coef[:,6]))

def calc_tau(v):
    ab = calc_alpha_beta(v)
    a = ab[...,::2]
    b = ab[...,1::2]
    return a / (a+b), 1.0 / (a+b)

def calc_index(v):
    return np.floor((v+0.5)*(maxVlt-minVlt)/SAMPLES + minVlt)

def v_range():
    return np.linspace(minVlt, maxVlt, SAMPLES)

def inf_tau_table():
    return calc_tau(v_range())

def write_chebyshev(names, data, deg=8):
    v = np.linspace(-1.0, 1.0, SAMPLES)
    deg = 8
    n = data.shape[1]
    p = np.zeros((9, n))

    for i in range(n):
        cheb = np.polynomial.chebyshev.Chebyshev.fit(v, data[:,i], deg)
        p[:,i] = cheb.coef
        s = '%s = %.5f' % (names[i], cheb.coef[0])
        for j in range(1, deg+1):
            f = cheb.coef[j]
            if j == 1:  # T1x2
                if f < 0:
                    s += ' - %.5f*T%dx2' % (np.abs(f)*0.5, j)
                elif f > 0:
                    s += ' + %.5f*T%dx2' % (f*0.5, j)
            elif j == deg-2:
                g = f - cheb.coef[j+2]
                if g < 0:
                    s += ' - %.5f*T%d' % (np.abs(g), j)
                elif g > 0:
                    s += ' + %.5f*T%d' % (g, j)
            elif j == deg:
                if f < 0:
                    s += ' - %.5f*T%da' % (np.abs(f), j)
                elif f > 0:
                    s += ' + %.5f*T%da' % (f, j)
            else:
                if f < 0:
                    s += ' - %.5f*T%d' % (np.abs(f), j)
                elif f > 0:
                    s += ' + %.5f*T%d' % (f, j)
        print(s)


def write_chebyshev_power(names, data, deg=8):
    v = np.linspace(-1.0, 1.0, SAMPLES)
    deg = 8
    n = data.shape[1]
    p = np.zeros((deg+1, n))
    q = np.zeros((deg+1, deg+1))
    q[0,0] = 1.0
    q[1,1] = 1.0
    for j in range(2, deg+1):
        q[1:,j] = 2 * q[:-1,j-1]
        q[:,j] -= q[:,j-2]

    q = np.transpose(q)

    for i in range(n):
        cheb = np.polynomial.chebyshev.Chebyshev.fit(v, data[:,i], deg)
        x = np.matmul(q, cheb.coef)
        s = '%s = %.5f ' % (names[i], x[0])
        left = 0
        for j in range(1, deg+1):
            s += '+ x*(%.5f' % x[j]
            left += 1

        s += ')' * left
        print(s)
