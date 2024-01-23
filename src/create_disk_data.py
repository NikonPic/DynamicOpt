# %%
from copy import copy
import numpy as np
from config import EnvSettings
from model_simulation import DiskModel, KneeModel, get_data, get_data_disk, get_segment_data_disk, get_segment_data_knee
import matplotlib.pyplot as plt
import json
import scipy.spatial.transform as sst
from scipy.spatial.transform import Rotation as R


def force_torque_signal(amplitude, frequency, phase, num_points, time_step):
    """
    Generate a force torque signal for simulation.

    Args:
    amplitude (tuple): Amplitudes for fx, fy, fz, tx, ty, tz.
    frequency (tuple): Frequencies for fx, fy, fz, tx, ty, tz.
    phase (tuple): Phases for fx, fy, fz, tx, ty, tz.
    num_points (int): Number of data points in the simulation.
    time_step (float): Time step for the simulation.

    Returns:
    np.array: 2D array containing the force torque signals for each component.
    """

    time = np.arange(0, num_points * time_step, time_step)
    force_torque = np.zeros((num_points, 6))

    for i in range(6):
        force_torque[:, i] = amplitude[i] * \
            np.sin(2 * np.pi * frequency[i] * time + phase[i])

    return force_torque


def stochastic_oscillator(omega, sigma, dt, N):
    X = np.zeros(N)
    for i in range(1, N):
        dW = np.sqrt(dt) * np.random.normal()
        X[i] = X[i-1] - omega**2 * X[i-1] * dt + sigma * dW
    return X


def force_torque_signal_rand(amplitude, frequency, phase, num_points, time_step):
    force_torque = np.zeros((num_points, 6))
    time = np.arange(0, num_points * time_step, time_step)

    for i in range(6):
        # Generate stochastic oscillators for amplitude and frequency
        amplitude_i = amplitude[i] + \
            stochastic_oscillator(0.1, 0.1, time_step, num_points)
        frequency_i = frequency[i] + \
            stochastic_oscillator(0.001, 0.01, time_step, num_points)

        # Generate the signal
        force_torque[:, i] = amplitude_i * \
            np.sin(2 * np.pi * frequency_i * time + phase[i])

    return force_torque


def quaternion_to_angular_velocity(q, dt):
    
    q = q.transpose()
    n = q.shape[1]
    omega = np.zeros((3, n))

    for t in range(1, n-1):
        # Use central differences for interior points
        q_prev = q[:, t-1]
        q_next = q[:, t+1]

        # Calculate the relative rotation from q_prev to q_next
        dq = sst.Rotation.from_quat(q_next).inv() * sst.Rotation.from_quat(q_prev)

        # Convert the quaternion difference to a rotation vector
        rot_vec = dq.as_rotvec()

        # Divide by 2*dt to get angular velocity
        omega[:, t] = rot_vec / (2 * dt)

    # Handle edge cases with forward/backward difference
    dq_first = sst.Rotation.from_quat(q[:, 1]).inv() * sst.Rotation.from_quat(q[:, 0])
    dq_last = sst.Rotation.from_quat(q[:, -1]).inv() * sst.Rotation.from_quat(q[:, -2])
    omega[:, 0] = dq_first.as_rotvec() / dt
    omega[:, -1] = dq_last.as_rotvec() / dt

    return omega.transpose()



def calculate_angular_velocities(qs, dt=0.01):
    rpos = quaternions_to_rotmats(qs.transpose())
    n = rpos.shape[2]
    angular_velocities = np.zeros((3, n))

    for t in range(n):
        if t == 0:  # Forward difference for the first element
            R_next = rpos[:, :, t + 1]
            R = rpos[:, :, t]
            diff = (R_next @ R.T - R @ R_next.T) / (dt)
        elif t == n - 1:  # Backward difference for the last element
            R_prev = rpos[:, :, t - 1]
            R = rpos[:, :, t]
            diff = (R @ R_prev.T - R_prev @ R.T) / (dt)
        else:  # Central difference for the rest
            R_next = rpos[:, :, t + 1]
            R_prev = rpos[:, :, t - 1]
            diff = (R_next @ R_prev.T - R_prev @ R_next.T) / (2 * dt)

        # Extract angular velocity from the skew-symmetric matrix
        angular_velocities[:, t] = np.array([diff[2, 1], diff[0, 2], diff[1, 0]])

    return angular_velocities.transpose()

def quaternions_to_rotmats(quaternions):
    n = quaternions.shape[1]
    rotmats = np.zeros((3, 3, n))

    for i in range(n):
        r = R.from_quat(quaternions[:, i])
        rotmats[:, :, i] = r.as_matrix()

    return rotmats


def calculate_angular_velocity_scipy(rotmat1, rotmat2, dt):
    # Create rotation objects
    r1 = R.from_matrix(rotmat1)
    r2 = R.from_matrix(rotmat2)

    # Compute the relative rotation from r1 to r2
    relative_rotation = r2 * r1.inv()

    # Convert to rotation vector (axis-angle representation)
    rot_vec = relative_rotation.as_rotvec()

    # The angular velocity is the rotation vector divided by time
    angular_velocity = rot_vec / dt

    return angular_velocity


def calculate_angular_velocities3(qs, dt=0.01):
    rpos = quaternions_to_rotmats(qs.transpose())
    n = rpos.shape[2]
    angular_velocities = np.zeros((3, n))

    for t in range(n):
        if t == 0:  # Forward difference for the first element
            R_next = R.from_matrix(rpos[:, :, t + 1])
            R_prev = R.from_matrix(rpos[:, :, t])
            relative_rotation = R_next * R_prev.inv()
            rot_vec = relative_rotation.as_rotvec()
            angular_velocity = rot_vec / dt

        elif t == n - 1:  # Backward difference for the last element
            R_next = R.from_matrix(rpos[:, :, t])
            R_prev = R.from_matrix(rpos[:, :, t-1])
            relative_rotation = R_next * R_prev.inv()
            rot_vec = relative_rotation.as_rotvec()
            angular_velocity = rot_vec / dt

        else:  # Central difference for the rest
            R_next = R.from_matrix(rpos[:, :, t + 1])
            R_prev = R.from_matrix(rpos[:, :, t-1])
            relative_rotation = R_next * R_prev.inv()
            rot_vec = relative_rotation.as_rotvec()
            angular_velocity = rot_vec / (2*dt)

        # Extract angular velocity from the skew-symmetric matrix
        angular_velocities[:, t] = angular_velocity

    return angular_velocities.transpose()


# %%
if __name__ == '__main__':
    cfg = EnvSettings(
        model_type='knee',
        path_ext='../models/knee/',
        scene_path='../models/knee/scene.xml',
        param_path='../models/knee',
        param_file='parameters_data_gen.yaml',
        crit_speed=20,
        max_episode_len=500,
        ft_input_len=6,
        freq=100,
        apply_ft_body_id=3,
        lam_reward=10,
        param_data_gen_file='',
        data_path = '',
    )

    t1 = 0000
    t2 = t1 + 500

    model = KneeModel(cfg)

    amplitude = (1, 1, .5, -1, .5, .15)
    frequency = (0.2, .1, .3, 2, .5, .4)
    phase = (0, 0, 0, 0, 0, 0)
    num_points = 100
    time_step = 0.01
    ft_ext = force_torque_signal_rand(
        amplitude, frequency, phase, num_points, time_step)
    qvel = model.simulate_dq(ft_ext=ft_ext, tend=int(num_points*time_step))

    # %%

    plt.figure()
    plt.plot(qvel[:num_points, :])
    
    
    qpos = model.simulate(ft_ext=ft_ext, tend=int(num_points*time_step))
    plt.figure()
    plt.plot(qpos[:num_points, :])
    
    # %%
    qvel2 = quaternion_to_angular_velocity(qpos, time_step)
    qvel3 = calculate_angular_velocities3(qpos, time_step)
    plt.figure()
    plt.plot(qvel[:num_points, :], '-')
    plt.gca().set_prop_cycle(None)
    #plt.plot(qvel2[:num_points, :], '--')
    plt.gca().set_prop_cycle(None)
    plt.plot(qvel3[:num_points, :], '-.')

    data_disk = {
        'ft': ft_ext.tolist(),
        'q': qpos.tolist()
    }

    with open('../measurement/data_knee_random.json', 'w') as f:
        json.dump(data_disk, f)

# %%
model = KneeModel(cfg)
data = get_data()
seg_data = get_segment_data_knee(data, 0, 500)


# %%
model.run(seg_data)
# %%
amplitude = (12, 3, 4, -.4, .3, .25)
frequency = (0.01, .1, .3, 2, .5, .4)
phase = (0, 0, 0, 0, 0, 0)
num_points = 2000
time_step = 0.01
ft_ext = force_torque_signal_rand(
    amplitude, frequency, phase, num_points, time_step)


qpos = model.simulate(ft_ext=ft_ext, tend=int(num_points*time_step))
plt.figure()
plt.plot(qpos[:num_points, :3])
plt.figure()
plt.plot(qpos[:num_points, 3:])
qvel = model.simulate_dq(ft_ext=ft_ext, tend=int(num_points*time_step))
plt.figure()
plt.plot(qvel)
# %%
tibia_qpos = copy(qpos)
tibia_qpos[:, :3] = 0
tibia_qpos[:, 3:] = 0
tibia_qpos[:, 3] = 1
s_app = copy(tibia_qpos[:, 3])
s_app = [25 for _ in range(len(s_app))]

data_knee = {
        'femur_pos': qpos[:, :3].tolist(),
        'femur_quat': qpos[:, 3:].tolist(),
        'tibia_pos': tibia_qpos[:, :3].tolist(),
        'tibia_quat': tibia_qpos[:, 3:].tolist(),
        'force_ext': ft_ext[:, :3].tolist(),
        'torque_ext': ft_ext[:, 3:].tolist(),
        's_app': s_app,
}
with open('../measurement/data_knee_random.json', 'w') as f:
    json.dump(data_knee, f)
# %%


def stochastic_oscillator(omega, sigma, dt, N):
    X = np.zeros(N)
    for i in range(1, N):
        dW = np.sqrt(dt) * np.random.normal()
        X[i] = X[i-1] - omega**2 * X[i-1] * dt + sigma * dW
    return X


def force_torque_signal(amplitude, frequency, phase, num_points, time_step):
    force_torque = np.zeros((num_points, 6))
    time = np.arange(0, num_points * time_step, time_step)

    for i in range(6):
        # Generate stochastic oscillators for amplitude and frequency
        amplitude_i = amplitude[i] + \
            stochastic_oscillator(0.1, 0.1, time_step, num_points)
        frequency_i = frequency[i] + \
            stochastic_oscillator(0.001, 0.01, time_step, num_points)

        # Generate the signal
        force_torque[:, i] = amplitude_i * \
            np.sin(2 * np.pi * frequency_i * time + phase[i])

    return force_torque


amplitude = (10, 10, 5, -10, 5, 15)
frequency = (0.2, .1, .3, 2, .5, .4)
phase = (0, 0, 0, 0, 0, 0)
num_points = 10000
time_step = 0.01

ft_ext = force_torque_signal(
    amplitude, frequency, phase, num_points, time_step)
plt.plot(ft_ext[4000:5000, :6])
# %%


def generate_signal(T, dt):
    t = np.arange(0, T, dt)

    # Base amplitude, frequency, and phase
    A0, F0, P0 = 1, 1, 0

    # Parameters for the first set of sinusoids
    A1, f_A1, phi_A1 = 0.5, 0.01, 0
    F1, f_F1, phi_F1 = 0.2, 0.005, 0
    P1, f_P1, phi_P1 = 0.1, 0.002, 0

    # Parameters for the second set of sinusoids
    A2, f_A2, phi_A2 = 0.25, 0.002, 0
    F2, f_F2, phi_F2 = 0.1, 0.001, 0
    P2, f_P2, phi_P2 = 0.05, 0.0005, 0

    # Time-varying amplitude, frequency, and phase
    A_t = A0 + A1 * np.sin(2 * np.pi * f_A1 * t + phi_A1) + \
        A2 * np.sin(2 * np.pi * f_A2 * t + phi_A2)
    F_t = F0 + F1 * np.sin(2 * np.pi * f_F1 * t + phi_F1) + \
        F2 * np.sin(2 * np.pi * f_F2 * t + phi_F2)
    P_t = P0 + P1 * np.sin(2 * np.pi * f_P1 * t + phi_P1) + \
        P2 * np.sin(2 * np.pi * f_P2 * t + phi_P2)

    # Generate the signal
    signal = A_t * np.sin(2 * np.pi * F_t * t + P_t)

    return t, signal


# Time parameters
T = 1000  # Total time
dt = 0.1  # Time step

# Generate and plot the signal
t, signal = generate_signal(T, dt)
plt.plot(t, signal)
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Time-Varying Sinusoidal Signal')
plt.show()

# %%
qpos
# %%
qvel.shape
# %%
