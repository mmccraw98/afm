import sympy as sp
import pickle

G = sp.Symbol('G')
T = sp.Symbol('Tau')
Gp = sp.Symbol('G_p')
Tp = sp.Symbol('Tau_p')
dt = sp.Symbol('\Delta t')
n = sp.Symbol('n')
s = sp.Symbol('s')
eps_0 = sp.Symbol('\epsilon_0')
Gp_min = sp.Symbol('G_p_min')
Gp_max = sp.Symbol('G_p_max')
Tp_min = sp.Symbol('T_p_min')
Tp_max = sp.Symbol('T_p_max')
t_max = sp.Symbol('T')
N = t_max / dt

t = n * dt
fp = eps_0 * t + eps_0 * Gp * Tp * (1 - sp.exp(-t / Tp)) - Gp * t * eps_0
f = eps_0 * t + eps_0 * G * T * (1 - sp.exp(-t / T)) - G * t * eps_0
offset = sp.log(2 * sp.pi * s ** 2) / 2
L_n = - offset - (f - fp) ** 2 / (2 * s ** 2)
L = sp.Sum(L_n, (n, 0, N))
t_integral = sp.integrate((L_n.diff(Gp) + L_n.diff(Tp)) ** 2, (Tp, Tp_min, Tp_max))
print('done integrating over G!')
g_integral = sp.integrate(t_integral, (Gp, Gp_min, Gp_max))
print('done integrating over T!')
integral = sp.Sum(g_integral, (n, 0, N))
with open('information_integral.pkl', 'wb') as f:
    pickle.dump(L, integral)
with open('information_integral.pkl.BACK', 'wb') as f:
    pickle.dump(L, integral)
    
