import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib import animation

# ==============================
# Funções auxiliares
# ==============================

def get_rotation_matrix(phi, theta, psi):
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])
    R_y = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    R_z = np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi), np.cos(psi), 0],
        [0, 0, 1]
    ])
    return R_z @ R_y @ R_x

def animate_quadcopter(positions, angles, l, save_gif=False, filename="quadcopter.gif"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    r = 0.1 * l
    ang = np.linspace(0, 2*np.pi, 50)
    propeller = np.vstack((r*np.cos(ang), r*np.sin(ang), np.zeros_like(ang))).T

    def update(frame):
        ax.clear()
        x, y, z = positions[frame]
        phi, theta, psi = angles[frame]
        R = get_rotation_matrix(phi, theta, psi)

        axle_x = np.array([[-l/2, 0, 0], [l/2, 0, 0]])
        axle_y = np.array([[0, -l/2, 0], [0, l/2, 0]])

        new_axle_x = (R @ axle_x.T).T + [x, y, z]
        new_axle_y = (R @ axle_y.T).T + [x, y, z]

        ax.plot(*new_axle_x.T, color='red', linewidth=2)
        ax.plot(*new_axle_y.T, color='black', linewidth=2)

        prop_ends = [new_axle_x[0], new_axle_x[1], new_axle_y[0], new_axle_y[1]]
        colors = ['r', 'g', 'b', 'c']

        for pos, color in zip(prop_ends, colors):
            prop_rotated = (R @ propeller.T).T + pos
            ax.plot(prop_rotated[:, 0], prop_rotated[:, 1], prop_rotated[:, 2], color=color)

        ax.set_xlim(x - l, x + l)
        ax.set_ylim(y - l, y + l)
        ax.set_zlim(z - l, z + l)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"t = {frame / len(positions):.2f}s")
        ax.view_init(elev=30, azim=120)

    ani = animation.FuncAnimation(fig, update, frames=len(positions), interval=30, blit=False)

    if save_gif:
        ani.save(filename, writer='pillow')
    else:
        plt.show()

    return ani

# ==============================
# Parâmetros
# ==============================

base_params = {
    'm': 0.468,
    'g': 9.81,
    'Ixx': 4.856e-3,
    'Iyy': 4.856e-3,
    'Izz': 8.801e-3,
    'l': 0.225,
    'K': 2.980e-6,
    'b': 1.14e-7,
    'Ax': 0.0,
    'Ay': 0.0,
    'Az': 0.0
}

omega = np.sqrt((base_params['m']*base_params['g']) / (4*base_params['K']))
speed = omega
dspeed = 0.1 * speed

omegas = {
    'omega1': speed,
    'omega2': speed,
    'omega3': speed,
    'omega4': speed
}

def eom_factory(params):
    def eom(t, Z):
        # desempacota
        x, y, z, phi, theta, psi, vx, vy, vz, phidot, thetadot, psidot = Z
        m, g = params['m'], params['g']
        Ixx, Iyy, Izz = params['Ixx'], params['Iyy'], params['Izz']
        l, K, b = params['l'], params['K'], params['b']
        Ax, Ay, Az = params['Ax'], params['Ay'], params['Az']
        o1, o2, o3, o4 = params['omega1'], params['omega2'], params['omega3'], params['omega4']

        # MATRIZ A (mesmo seu esquema)
        A = np.zeros((6,6))
        A[0,0]=A[1,1]=A[2,2]=m
        A[3,3]= Ixx; A[3,5]= -Ixx*np.sin(theta)
        A[4,4]= Iyy - Iyy*np.sin(phi)**2 + Izz*np.sin(phi)**2
        A[4,5]= np.cos(phi)*np.cos(theta)*np.sin(phi)*(Iyy - Izz)
        A[5,3]= -Ixx*np.sin(theta)
        A[5,4]= np.cos(phi)*np.cos(theta)*np.sin(phi)*(Iyy - Izz)
        A[5,5]= (Ixx*np.sin(theta)**2 +
                 Izz*np.cos(phi)**2*np.cos(theta)**2 +
                 Iyy*np.cos(theta)**2*np.sin(phi)**2)

        # VETOR B
        B = np.zeros((6,1))
        o_sum = o1**2 + o2**2 + o3**2 + o4**2
        # forças
        B[0,0] = K*( np.sin(phi)*np.sin(psi)
                   + np.cos(phi)*np.cos(psi)*np.sin(theta) )*o_sum - Ax*vx
        B[1,0] = -Ay*vy - K*( np.cos(psi)*np.sin(phi)
                             - np.cos(phi)*np.sin(psi)*np.sin(theta) )*o_sum
        B[2,0] = K*np.cos(phi)*np.cos(theta)*o_sum - m*g - Az*vz
        # momentos de controle
        tau_phi   = l*K*(o2**2 - o4**2)             # roll
        tau_theta = l*K*(o3**2 - o1**2)             # pitch
        tau_psi   = b*(o1**2 - o2**2 + o3**2 - o4**2)# yaw
        B[3,0] = tau_phi
        B[4,0] = tau_theta
        B[5,0] = tau_psi

        # resolve acelerações
        acc = np.linalg.solve(A, B).flatten()
        ax, ay, az, alphap, alphat, alphapsi = acc

        # derivadas de estado
        return np.array([
            vx, vy, vz,
            phidot, thetadot, psidot,
            ax, ay, az,
            alphap, alphat, alphapsi
        ])
    return eom

# ==============================
# Variações de m e l
# ==============================

fatores = {
    'original':        (1.0, 1.0),
    'm+20%':           (1.2, 1.0),
    'm-20%':           (0.8, 1.0),
    'l+20%':           (1.0, 1.2),
    'l-20%':           (1.0, 0.8),
}

resultados = {}

for label, (fm, fl) in fatores.items():
    params = base_params.copy()
    params.update(omegas)
    params['m'] = base_params['m'] * fm
    params['l'] = base_params['l'] * fl

    # roda simulação
    Z0 = np.zeros(12)
    Z0[3] = np.deg2rad(5)   # φ inicial = 5°
    Z0[4] = np.deg2rad(3)   # θ inicial = 3°
    t_span = (0, 1)
    t_eval = np.linspace(*t_span, 500)
    sol = solve_ivp(eom_factory(params), t_span, Z0, t_eval=t_eval,
                    args=(), rtol=1e-12, atol=1e-12)
    resultados[label] = sol

# Exemplo: se for cenário “m-20%” eu gero um pitch ativo
if label == 'm-20%':
    params['omega1'] += 20    # hélice 1 mais rápida
    params['omega3'] -= 20    # hélice 3 mais lenta

# ==============================
# Plot comparativo
# ==============================

# definição de estilos pra cada caso
styles = {
    'original': dict(color='blue',    linestyle='-'),
    'm+20%':    dict(color='orange',  linestyle='--'),
    'm-20%':    dict(color='green',   linestyle=':'),
    'l+20%':    dict(color='red',     linestyle='--'),
    'l-20%':    dict(color='purple',  linestyle=(0, (3,1,1,1)))
}

# 1) Posições x, y, z
plt.figure(figsize=(10, 6))
for coord, idx in zip(['x','y','z'], [0,1,2]):
    plt.subplot(3,1,idx+1)
    for label, sol in resultados.items():
        plt.plot(sol.t, sol.y[idx],
                 label=label,
                 **styles[label])
    plt.ylabel(coord)
    if idx==0: plt.title("Comparação de Posições")
    if idx==2: plt.xlabel("Tempo (s)")
    plt.grid(True)
    plt.legend(fontsize=8)
plt.tight_layout()
plt.show()


# 2) Ângulos φ, θ, ψ
plt.figure(figsize=(10, 6))
for ang, idx in zip([r'$\phi$', r'$\theta$', r'$\psi$'], [3,4,5]):
    plt.subplot(3,1,idx-2)
    for label, sol in resultados.items():
        plt.plot(sol.t, sol.y[idx], label=label)
    plt.ylabel(ang)
    if idx==3: plt.title("Comparação de Ângulos de Euler")
    if idx==5: plt.xlabel("Tempo (s)")
    plt.grid(True)
    plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# ==============================
# Animação
# ==============================
pos0 = resultados['original'].y[0:3, :].T
ang0 = resultados['original'].y[3:6, :].T
animate_quadcopter(pos0, ang0, l=base_params['l'], save_gif=False)

print('Fim da simulação comparativa.')