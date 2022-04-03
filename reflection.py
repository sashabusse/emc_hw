import numpy as np
import units_convertion as uconv


def angle_fall2pass(alpha_fall, n_fall, n_pass):
    # преобразование угла падения в преломление
    return np.arcsin(np.sin(alpha_fall) * n_fall/n_pass)


def Rh(alpha_fall, mu_fall=1., eps_fall=1., mu_pass=1., eps_pass=1., polarization='in_plane'):
    # Rh = Href/Hfall
    # polarization = 'in_plane'/'out_of_plane'
    # амплитудный коэффициент отражения для вектора напряженности магнитного поля
    p_fall = uconv.wave_res(mu_fall, eps_fall)
    p_pass = uconv.wave_res(mu_pass, eps_pass)
    n_fall = uconv.ref_index(mu_fall, eps_fall)
    n_pass = uconv.ref_index(mu_pass, eps_pass)

    alpha_pass = angle_fall2pass(alpha_fall, n_fall, n_pass)

    if polarization == 'out_of_plane':
        return (p_fall*np.cos(alpha_fall) - p_pass*np.cos(alpha_pass)) / \
            (p_fall*np.cos(alpha_fall) + p_pass*np.cos(alpha_pass))
    elif polarization == 'in_plane':
        return (p_pass*np.cos(alpha_fall) - p_fall*np.cos(alpha_pass)) / \
            (p_pass*np.cos(alpha_fall) + p_fall*np.cos(alpha_pass))
    else:
        assert False, 'bad polarization value: {}'.format(polarization)


def Th(alpha_fall, mu_fall=1., eps_fall=1., mu_pass=1., eps_pass=1., polarization='in_plane'):
    # Th = Hpass/Hfall = 1+Rh
    # амплитудный коэффициент прохождения для вектора напряженности магнитного поля
    p_fall = uconv.wave_res(mu_fall, eps_fall)
    p_pass = uconv.wave_res(mu_pass, eps_pass)

    if polarization == 'out_of_plane':
        return 1 + Rh(alpha_fall, mu_fall, eps_fall, mu_pass, eps_pass, polarization)
    elif polarization == 'in_plane':
        return (1+Rh(alpha_fall, mu_fall, eps_fall, mu_pass, eps_pass, polarization)) * p_fall/p_pass
    else:
        assert False, 'bad polarization value: {}'.format(polarization)


def Rh2Rp(rh):
    # преобразование амплитудного коэффициента отражения в коэффициент отражения по мощности
    return np.abs(rh)**2


def Th2Tp(th, alpha_fall, mu_fall=1., eps_fall=1., mu_pass=1., eps_pass=1.):
    # преобразование амплитудного коэффициента прохождения в коэффициент прохождения по мощности
    p_fall = uconv.wave_res(mu_fall, eps_fall)
    p_pass = uconv.wave_res(mu_pass, eps_pass)
    n_fall = uconv.ref_index(mu_fall, eps_fall)
    n_pass = uconv.ref_index(mu_pass, eps_pass)
    alpha_pass = angle_fall2pass(alpha_fall, n_fall, n_pass)
    return np.abs(th)**2 * (p_pass*np.cos(alpha_pass))/(p_fall*np.cos(alpha_fall))


def Rp(alpha_fall, mu_fall=1., eps_fall=1., mu_pass=1., eps_pass=1., polarization='in_plane'):
    # power reflection
    return Rh2Rp(Rh(alpha_fall, mu_fall, eps_fall, mu_pass, eps_pass, polarization))


def Tp(alpha_fall, mu_fall=1., eps_fall=1., mu_pass=1., eps_pass=1., polarization='in_plane'):
    # power transmission
    return Th2Tp(Th(alpha_fall, mu_fall, eps_fall, mu_pass, eps_pass, polarization), alpha_fall, mu_fall, eps_fall, mu_pass, eps_pass)


def HE_mat(w, l, mu=1., eps=1., alpha=0):
    k = uconv.wave_num(w, mu, eps)
    p = uconv.wave_res(mu, eps)
    return np.array([[np.cos(k*l*np.cos(alpha)),      -np.sin(k*l*np.cos(alpha))*(1j/p)],
                     [-1j*p*np.sin(k*l*np.cos(alpha)), np.cos(k*l)*np.cos(alpha)]], dtype=complex)


def HE_mat_RhTh(m, mu_fr=1., eps_fr=1., mu_to=1., eps_to=1.):
    p_fr = uconv.wave_res(mu_fr, eps_fr)
    p_to = uconv.wave_res(mu_to, eps_to)

    assert abs(m[1, 1]-m[0, 0]) < 1e-6, 'm22 should be equal to m22'

    denominator = (m[0, 0] + m[0, 1]*p_to)*p_fr + (m[1, 0] + m[1, 1]*p_to)

    rh = ((m[0, 0] + m[0, 1]*p_to)*p_fr -
          (m[1, 0] + m[1, 1]*p_to)) / denominator

    th = 2*p_fr / denominator

    return rh, th
