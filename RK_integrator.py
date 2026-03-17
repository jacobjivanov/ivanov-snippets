import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600

A = np.array([
    [0, 0, 0, 0],
    [1/3, 0, 0, 0],
    [-1/3, 1, 0, 0],
    [1, -1, 1, 0]
])
B = np.array([1/8, 3/8, 3/8, 1/8])
C = np.array([0, 1/3, 2/3, 1])

def rk_step(f, t0, y0, h, A, B, C, *args):
    """
    Generic explicit Runge-Kutta step.
    Works for scalar or vector-valued y0.
    Non-autonomous ODE: y' = f(t, y)

    Parameters
    ----------
    f  : function f(t, y)
    t0 : current time
    y0 : current state (scalar or vector)
    h  : step size
    A, B, C : Butcher tableau arrays
              A = stage matrix
              B = weights
              C = nodes
    """
    s = len(B)                 # number of stages
    y0 = np.asarray(y0)

    k = np.zeros((s,) + y0.shape)

    for i in range(s):
        t_i = t0 + C[i] * h     # stage time

        y_i = y0.copy()
        for j in range(i):
            y_i += h * A[i, j] * k[j]

        k[i] = f(t_i, y_i, *args)

    y1 = y0.copy()
    for i in range(s):
        y1 += h * B[i] * k[i]

    return y1

def f(t, y):
    return np.exp(-t) - y

def solution(t, y0):
    y = (y0 + t) * np.exp(-t)
    return y

def INTEGRATE(f, t0, t1, y0, N):
    t = np.linspace(t0, t1, N)
    y = np.empty(N)
    y[0] = y0

    for i in range(1, len(t)):
        y[i] = rk_step(f, t[i-1], y[i-1], t[1], A, B, C)

    return t, y

def ERROR_inf(t0, t1, y, y0):
    N = y.size
    t = np.linspace(t0, t1, N)

    error = solution(t, y0) - y

    return np.max(np.abs(error))

t0 = 0.
t1 = 20.

y0 = np.array([-5, -3, -1, -0.1, 0, 0.1, 1, 3, 5])
N = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000, 250000])
error_inf = np.empty(shape = (len(y0), len(N)))

for j in range(0, len(y0)):
    for i in range(0, len(N)):
        t, y = INTEGRATE(f, t0, t1, y0[j], N[i])

        error_inf[j, i] = ERROR_inf(t0, t1, y, y0[j])

for j in range(0, len(y0)):
    plt.loglog(N, error_inf[j], marker = 'x', linestyle = 'none', label = y0[j])
plt.gca().set(ylim = (1e-16, 1e1), xlim = (1e0, 1e6), ylabel = r'$\ell_\infty$ Error', title = r"Runge-Kutta Convergence of $y' = \exp (-t) - y$")
plt.legend()
plt.grid()
plt.show()