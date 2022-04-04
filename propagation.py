import numpy as np
import units_convertion as uconv
from matplotlib import pyplot as plt
from scipy.integrate import quad


def free_space_loss(lamb, R):
    return (lamb/(4*np.pi*R))**2


def free_space_loss_db(lamb, R):
    return uconv.P2db(free_space_loss(lamb, R))


def rain_attenuation_db(f, R, k, J, polarization='vertical'):
    if polarization == 'vertical':
        cfg = {
            'a': [-2.125, 16.48, -87.9, 232.2],
            'b': [-12.39, 4.1, -0.288]
        }
    elif polarization == 'horizontal':
        cfg = {
            'a': [-1.761, 13.81, -62.77, 142],
            'b': [-12.76, 4.365, -0.324]
        }
    else:
        assert False, 'bad polarization string: {}'.format(polarization)

    lnf = np.log(f)
    a = cfg['a'][0] + cfg['a'][1]/lnf + \
        cfg['a'][2]/(lnf**3) + cfg['a'][3]/(lnf**5)
    b = cfg['b'][0] + cfg['b'][1]*lnf + cfg['b'][2]*(lnf**2)

    print("a={}, b={}".format(a, b))

    gd = b*(J**a)
    print(gd)

    R_ef = (R*1e-3)*k
    return gd*R_ef


def oxygen_attenuation_db_per_km(f_hz):
    f_ghz = f_hz * 1e-9
    return (7.19e-3 +
            6.09/(f_ghz**2+0.227) +
            4.81/((f_ghz-57)**2 + 1.5)) * f_ghz**2 * 1e-3


def h20_attenuation_db_per_km(f_hz, p_h20):
    f_ghz = f_hz*1e-9
    return (0.05 +
            0.0021*p_h20 +
            3.6 / ((f_ghz-22.2)**2 + 8.5) +
            10.6/((f_ghz-183.3)**2 + 9) +
            8.9 / ((f_ghz-325.4)**2 + 26.3))*(f_ghz**2)*p_h20*1e-4


def atmospher_attenuation_db_per_km(f_hz, p_h20, temperature, print_en=False):
    g_ox = oxygen_attenuation_db_per_km(f_hz)
    g_h20 = h20_attenuation_db_per_km(f_hz, p_h20)
    g = (1-(temperature-15)*0.01)*g_ox + \
        (1-(temperature-15)*0.06)*g_h20

    if print_en:
        print("atmospher attenuation calculation:")
        print("\tf={:.3f} GHz, t={}, p_h20={}".format(
            f_hz*1e-9, temperature, p_h20))
        print("\tg_ox = {} dB/km".format(g_ox))
        print("\tg_h20 = {} dB/km".format(g_h20))
        print("\tg = {} dB/km".format(g))

    return g


def point2line_distance(lx1, ly1, lx2, ly2, px, py):
    dx1 = lx1-px
    dx2 = lx2-px
    dy1 = ly1-py
    dy2 = ly2-py

    l = np.sqrt((lx1-lx2)**2+(ly1-ly2)**2)

    S = dx2*dy1 - dx1*dy2
    return S/l


def re_refraction(refraction, print_en=False):
    Re0 = 6371e3
    Re_eq = Re0/(1+Re0*refraction/2)
    if print_en:
        print("Earth radius equivalent calculation:")
        print("\tRe_eq={} km".format(Re_eq*1e-3))
    return Re_eq


def track_earth_lvl(Rt, refraction, print_en=False):
    Re_eq = re_refraction(refraction, print_en)
    R0 = Rt[-1]
    y0 = (R0**2)/(2*Re_eq)*(Rt/R0)*(1-Rt/R0)
    return y0


def track_clearance(Rt, yt, H_tx, H_rx, refraction, print_en=False):
    y0 = track_earth_lvl(Rt, refraction, print_en)
    y = y0 + yt
    y_tx = y[0] + H_tx
    y_rx = y[-1] + H_rx

    h_clr = np.zeros(y.size)
    for i in range(h_clr.size):
        h_clr[i] = point2line_distance(Rt[0], y_tx, Rt[-1], y_rx, Rt[i], y[i])

    return (y, y_tx, y_rx, h_clr)


def plot_clearance(Rt, y, y_tx, y_rx, h_clr, refraction):
    y0 = track_earth_lvl(Rt, refraction)

    plt.figure(figsize=(9, 5))

    plt.plot(Rt, y0, 'r-', label='Earth curvature')

    plt.plot(Rt, y, 'b-', label='ground')
    plt.scatter([Rt[0], Rt[-1]], [y_tx, y_rx], label='RX/TX')
    plt.plot([Rt[0], Rt[-1]], [y_tx, y_rx], 'r--', label='TX-RX')
    for i in range(Rt.size):
        plt.text(Rt[i], y[i], '{:.2f}'.format(h_clr[i]))

    plt.xlabel('Rt м')
    plt.ylabel('h м')
    plt.grid(True)
    plt.legend()


def V_interf_attenuation_db(PHI, refraction, f, R0, R_clr, H_clr, print_en=False):
    k = R_clr/R0
    H0 = np.sqrt(R0*uconv.f2lamb(f)*k*(1-k)/3)
    dH = -(R0**2)*refraction*k*(1-k)/4
    p = (H_clr+dH)/H0
    V = np.sqrt(1 + (PHI**2) - 2*PHI*np.cos(np.pi*(p**2)/3))

    if print_en:
        print("V_interf_attenuation calculation:")
        print("\tdH = {} m".format(dH))
        print("\tH0 = {} m".format(H0))
        print("\tp(ref) = {}".format(p))
        print("\tV_interf = {} dB".format(V))

    return V


def optical_spiral(v):
    a = quad(lambda x: np.cos(np.pi*(x**2)/2), 0, v)[0]
    b = quad(lambda x: np.sin(np.pi*(x**2)/2), 0, v)[0]

    return (0.5 + a)/np.sqrt(2), (0.5 + b)/np.sqrt(2)


def normalized_clearance(H, R1, R2, lamb):
    return H*np.sqrt(2*(R1+R2)/R1/R2)


def obstacle_attenuation_curve_db(u):
    P = np.zeros(u.size)
    for i in range(u.size):
        a, b = optical_spiral(u[i])
        P[i] = (np.abs(a)**2 + np.abs(b**2))

    return uconv.P2db(P)


def obstacles_attenuation_db(R, h, lamb, print_en=False):
    if print_en:
        print("calculating attenuation on obstacles:")

    res = 0
    for i in range(1, R.size-1):
        hl = h[i-1] + (h[i+1] - h[i-1]) * (R[i]-R[i-1])/(R[i+1]-R[i-1])
        H = hl-h[i]
        R1 = R[i]-R[i-1]
        R2 = R[i+1]-R[i]
        norm_clr = normalized_clearance(H, R1, R2, lamb)
        attenuation_db = obstacle_attenuation_curve_db(np.array(
            [norm_clr]
        ))
        res += attenuation_db
        if print_en:
            print("\t {}th obstacle:".format(i))
            print("\t\tH = {}".format(H))
            print("\t\tR1 = {}".format(R1))
            print("\t\tR2 = {}".format(R2))
            print("\t\tnormalized clearance = {}".format(norm_clr))
            print("\t\tattenuation = {} dB".format(attenuation_db))
            print()

    print("total attenuation = {} dB".format(res))
    return res


def obstacles_H0_attenuation_db(R0, R1, R2, R3):
    V = np.abs(0.5*(1-np.arctan(np.sqrt(R2*R0/R1/R3))/np.pi))
    return uconv.P2db(V)
