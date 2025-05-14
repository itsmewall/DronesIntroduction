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

        # Axles in body frame
        axle_x = np.array([[-l/2, 0, 0], [l/2, 0, 0]])
        axle_y = np.array([[0, -l/2, 0], [0, l/2, 0]])

        # Transform to world frame
        new_axle_x = (R @ axle_x.T).T + [x, y, z]
        new_axle_y = (R @ axle_y.T).T + [x, y, z]

        # Draw arms
        ax.plot(*new_axle_x.T, color='red', linewidth=2)
        ax.plot(*new_axle_y.T, color='black', linewidth=2)

        # Propeller positions: 4 ends
        prop_ends = [new_axle_x[0], new_axle_x[1], new_axle_y[0], new_axle_y[1]]
        colors = ['r', 'g', 'b', 'c']

        # Draw each propeller
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
# Parâmetros e EDO
# ==============================

params = {
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

omega = 1.5
speed = omega * np.sqrt(1 / params['K'])
dspeed = 0.1 * speed
params['omega1'] = speed - dspeed
params['omega2'] = speed - dspeed
params['omega3'] = speed 
params['omega4'] = speed

def eom(t, Z):
    m, g = params['m'], params['g']
    Ixx, Iyy, Izz = params['Ixx'], params['Iyy'], params['Izz']
    l, K, b = params['l'], params['K'], params['b']
    Ax, Ay, Az = params['Ax'], params['Ay'], params['Az']
    omega1, omega2, omega3, omega4 = params['omega1'], params['omega2'], params['omega3'], params['omega4']
    
    x, y, z, phi, theta, psi, vx, vy, vz, phidot, thetadot, psidot = Z
    Tz = K * (omega1**2 + omega2**2 + omega3**2 + omega4**2)



    A = np.zeros((6, 6))

    A[0, 0] = m
    A[1, 1] = m
    A[2, 2] = m
    A[3, 3] = Ixx
    A[3, 5] = -Ixx * np.sin(theta)
    A[4, 4] = Iyy - Iyy * np.sin(phi)**2 + Izz * np.sin(phi)**2
    A[4, 5] = np.cos(phi) * np.cos(theta) * np.sin(phi) * (Iyy - Izz)
    A[5, 3] = -Ixx * np.sin(theta)
    A[5, 4] = np.cos(phi) * np.cos(theta) * np.sin(phi) * (Iyy - Izz)
    A[5, 5] = (Ixx * np.sin(theta)**2 + 
            Izz * np.cos(phi)**2 * np.cos(theta)**2 + 
            Iyy * np.cos(theta)**2 * np.sin(phi)**2)

    # Now the B vector
    B = np.zeros((6, 1))

    omega_sum = omega1**2 + omega2**2 + omega3**2 + omega4**2
    cos_phi = np.cos(phi)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    sin_theta = np.sin(theta)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    B[0, 0] = K * (sin_phi * sin_psi + cos_phi * cos_psi * sin_theta) * omega_sum - Ax * vx
    B[1, 0] = -Ay * vy - K * (cos_psi * sin_phi - cos_phi * sin_psi * sin_theta) * omega_sum
    B[2, 0] = K * cos_phi * cos_theta * omega_sum - g * m - Az * vz

    B[3, 0] = ((Izz * thetadot**2 * np.sin(2*phi)) / 2 
            - (Iyy * thetadot**2 * np.sin(2*phi)) / 2 
            - K * l * omega2**2 + K * l * omega4**2 
            + Ixx * psidot * thetadot * cos_theta 
            - Iyy * psidot * thetadot * cos_theta 
            + Izz * psidot * thetadot * cos_theta 
            + Iyy * psidot**2 * cos_phi * cos_theta**2 * sin_phi 
            - Izz * psidot**2 * cos_phi * cos_theta**2 * sin_phi 
            + 2 * Iyy * psidot * thetadot * cos_phi**2 * cos_theta 
            - 2 * Izz * psidot * thetadot * cos_phi**2 * cos_theta)

    B[4, 0] = ((Ixx * psidot**2 * np.sin(2*theta)) / 2 
            - K * l * omega1**2 + K * l * omega3**2 
            - Ixx * phidot * psidot * cos_theta 
            + Iyy * phidot * thetadot * np.sin(2*phi) 
            - Izz * phidot * thetadot * np.sin(2*phi) 
            - Izz * psidot**2 * cos_phi**2 * cos_theta * sin_theta 
            - Iyy * psidot**2 * cos_theta * sin_phi**2 * sin_theta 
            - Iyy * phidot * psidot * cos_phi**2 * cos_theta 
            + Izz * phidot * psidot * cos_phi**2 * cos_theta 
            + Iyy * phidot * psidot * cos_theta * sin_phi**2 
            - Izz * phidot * psidot * cos_theta * sin_phi**2)

    B[5, 0] = (b * omega1**2 - b * omega2**2 + b * omega3**2 - b * omega4**2 
            + Ixx * phidot * thetadot * cos_theta 
            + Iyy * phidot * thetadot * cos_theta 
            - Izz * phidot * thetadot * cos_theta 
            - Ixx * psidot * thetadot * np.sin(2*theta) 
            + Iyy * psidot * thetadot * np.sin(2*theta) 
            + Iyy * thetadot**2 * cos_phi * sin_phi * sin_theta 
            - Izz * thetadot**2 * cos_phi * sin_phi * sin_theta 
            - 2 * Iyy * phidot * thetadot * cos_phi**2 * cos_theta 
            + 2 * Izz * phidot * thetadot * cos_phi**2 * cos_theta 
            - 2 * Iyy * phidot * psidot * cos_phi * cos_theta**2 * sin_phi 
            + 2 * Izz * phidot * psidot * cos_phi * cos_theta**2 * sin_phi 
            - 2 * Iyy * psidot * thetadot * cos_phi**2 * cos_theta * sin_theta 
            + 2 * Izz * psidot * thetadot * cos_phi**2 * cos_theta * sin_theta)




    acc = np.linalg.solve(A, B).flatten()

    return np.array([vx, vy, vz, phidot, thetadot, psidot, *acc])

# ==============================
# Simulação
# ==============================

Z0 = np.zeros(12)

t_span = (0, 1)
t_eval = np.linspace(*t_span, 500)

sol = solve_ivp(eom, t_span, Z0, t_eval=t_eval, rtol=1e-12, atol=1e-12)

# ==============================
# Plot dos resultados
# ==============================

plt.figure()
labels = ['x', 'y', 'z', r'$\phi$', r'$\theta$', r'$\psi$',
          r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$',
          r'$\dot{\phi}$', r'$\dot{\theta}$', r'$\dot{\psi}$']
for i in range(12):
    plt.plot(sol.t, sol.y[i], label=labels[i])
plt.legend(fontsize=10)
plt.xlabel('Tempo (s)')
plt.ylabel('Estados')
plt.title('Simulação do Quadricóptero')
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# Animação
# ==============================

positions = sol.y[0:3, :].T
angles = sol.y[3:6, :].T
animate_quadcopter(positions, angles, l=params['l'], save_gif=False)
print('end')
