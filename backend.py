from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

T = 100000  # ms
lambd = 80  # Hz
N_exc = 8
N_inh = 2
N_out = 2
N_x = 10
V_reset = -80
N = N_x + N_out
Rm = 10
Cm = 1
tau_z = 50
max_diff = 100
min_diff = -100
N_items = 100
A_plus = 0.05
A_minus = 0.04
tau_plus = 100
tau_minus = 100
exc_weights = 1.5
inh_weights = -1.5
U_rest = -70
learning_rate = 0.05
V_thresh = -55
plot_data_1 = False
plot_data_2 = True
anti_STDP_ = True
anti_hebb_ = False
STDP_ = False

weights = np.zeros(shape=(N_x, T))
weights[:8, 0] = exc_weights
weights[8:, 0] = inh_weights

float_data = np.zeros(shape=(N_items, N_x))

# Parameters
duration = 1000  # Total simulation time in ms
dt = 1  # Time step in ms

# Derived values
p_spike = lambd / T  # Spike probability per time step

# Generate binary spike trains
spike_matrix = (np.random.rand(N_x, T) < p_spike).astype(int)
new_stack = np.zeros(shape=(2, T))
spike_matrix = np.vstack([spike_matrix, new_stack])

prev_S = np.zeros(shape=(N))
prev_S = np.random.randint(low=1000, high=100000, size=N)
z_trace = np.zeros(shape=(N, T))

# spike_matrix now contains 1s for spikes and 0s for no spikes
# Extract spike times and neuron indices for plotting
if plot_data_1:
    positions = [np.where(spike_matrix[n, :] == 1)[0] for n in range(N_x)]
    colors = [
        "green",
        "green",
        "green",
        "green",
        "green",
        "green",
        "green",
        "green",
        "red",
        "red",
    ]
    plt.eventplot(positions=positions, colors=colors)
    plt.savefig("/home/andreas-massey/Documents/FYS_article/training_spikes.png")
    plt.show()


U = np.full(shape=(N_out, T), fill_value=U_rest)


def anti_STDP(prev_S, weights, A_plus, A_minus, tau_plus, tau_minus):
    delta_t = prev_S[-1] - prev_S[:-1]

    for i in range(len(delta_t)):
        if delta_t[i] > max_diff or delta_t[i] < min_diff:
            continue
        if delta_t[i] > 0:
            delta_w = -A_minus * np.exp(delta_t[i] / tau_minus)
            weights[i] = min(-0.1, delta_w)
        else:
            delta_w = A_plus * np.exp(-delta_t[i] / tau_plus)
            weights[i] = min(-0.1, delta_w)
    return weights


def anti_hebb(z_trace, learning_rate, weights):
    weights += np.multiply(z_trace[-2:], z_trace[:-2]) * learning_rate
    return weights


def STDP(prev_S, weights, A_plus, A_minus, tau_plus, tau_minus):
    delta_t = prev_S[-1] - prev_S[:-1]

    for i in range(len(delta_t)):
        if delta_t[i] > max_diff or delta_t[i] < min_diff:
            continue
        if delta_t[i] > 0:
            delta_w = A_plus * np.exp(-delta_t[i] / tau_plus)
            weights[i] = min(delta_w, -0.01)
        else:
            delta_w = -A_minus * np.exp(delta_t[i] / tau_minus)
            weights[i] = min(-0.01, delta_w)
    return weights


import matplotlib.pyplot as plt

method_names = ["Anti-Hebbian", "Anti-STDP", "STDP"]
colors = ["red", "blue", "green"]

# Data collection for analysis
all_positions = []  # Spiking activity
all_positions_inh_exc = []  # Inhibitory vs. excitatory spiking
all_potentials = []  # Membrane potential
all_cross_corrs = []  # Cross-correlation
all_weights = []  # To store weights for each method
all_spikes = []

for j in range(3):
    # Configure the learning rule for this iteration
    if j == 0:
        anti_hebb_ = True
        anti_STDP_ = False
        STDP_ = False
    elif j == 1:
        anti_hebb_ = False
        anti_STDP_ = True
        STDP_ = False
    elif j == 2:
        anti_hebb_ = False
        anti_STDP_ = False
        STDP_ = True
    for t in tqdm(range(1, T)):
        # update membrane potential
        I_in = np.dot(weights[:, t - 1], spike_matrix[:-N_out, t - 1])
        U_delta = (
            I_in
            - ((U[0, t - 1] - U_rest) * (V_thresh - U[0, t - 1]))
            / (Rm * (V_thresh - U_rest))
        ) / Cm
        U[0, t] = U[0, t - 1] + U_delta

        I_in = np.dot(weights[:-N_inh, t - 1], spike_matrix[: -(N_inh + N_out), t - 1])
        U_delta = (
            I_in
            - ((U[1, t - 1] - U_rest) * (V_thresh - U[1, t - 1]))
            / (Rm * (V_thresh - U_rest))
        ) / Cm
        U[1, t] = U[1, t - 1] + U_delta

        # update trace
        z_trace[:, t] += -z_trace[:, t - 1] / tau_z

        # update spike_timing
        for n in range(1, N_out + 1):
            if U[n - 1, t] > V_thresh:
                spike_matrix[-n, t] = 1
                U[n - 1, t] = V_reset

        prev_S = np.where(spike_matrix[:, t] == 1, 0, prev_S + 1)
        z_trace[:, t][spike_matrix[:, t] == 1] += 1

        # update weights
        if anti_STDP_:
            weights[N_exc:, t] = anti_STDP(
                prev_S=prev_S[N_exc:-N_out],
                weights=weights[N_exc:, t - 1],
                A_plus=A_plus,
                A_minus=A_minus,
                tau_plus=tau_plus,
                tau_minus=tau_minus,
            )
            weights[:N_exc, t] = weights[:N_exc, t - 1]
        elif anti_hebb_:
            weights[N_exc:, t] = anti_hebb(
                weights=weights[N_exc:, t - 1],
                z_trace=z_trace[N_exc:, t - 1],
                learning_rate=learning_rate,
            )
            weights[:N_exc, t] = weights[:N_exc, t - 1]
        elif STDP_:
            weights[N_exc:, t] = STDP(
                prev_S=prev_S[N_exc:-N_out],
                weights=weights[N_exc:, t - 1],
                A_plus=A_plus,
                A_minus=A_minus,
                tau_plus=tau_plus,
                tau_minus=tau_minus,
            )
            weights[:N_exc, t] = weights[:N_exc, t - 1]
    all_weights.append(weights.copy())
    all_spikes.append(spike_matrix.copy())

    # Collect data for spiking activity
    positions = [np.where(spike_matrix[n, :] == 1)[0] for n in range(N)]
    all_positions.append(positions)

    # Collect data for inhibitory vs. excitatory spiking patterns
    positions = [np.where(spike_matrix[n, :] == 1)[0] for n in range(N_exc, N)]
    all_positions_inh_exc.append(positions)

    # Collect membrane potential data
    all_potentials.append(np.transpose(U))

    # Compute and collect cross-correlation
    lags = np.arange(-T + 1, T)
    cross_corr = correlate(
        spike_matrix[-N_out], spike_matrix[-(N_out + N_inh)], mode="full"
    )
    cross_corr_normalized = cross_corr / np.sqrt(
        np.sum(spike_matrix[-N_out] ** 2) * np.sum(spike_matrix[-(N_out + N_inh)] ** 2)
    )
    all_cross_corrs.append((lags, cross_corr_normalized))

import matplotlib.pyplot as plt

# Define method names and colors for comparison
method_names = ["Anti-Hebbian", "Anti-STDP", "STDP"]
colors = ["red", "blue", "green"]

# Plot 1: Spiking Activity
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for j in range(3):
    axs[j].eventplot(all_positions[j], colors=[colors[j]] * len(all_positions[j]))
    axs[j].set_title(f"{method_names[j]}")
    axs[j].set_xlabel("Time")
    axs[j].set_ylabel("Neuron Index")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Plot 2: Inhibitory vs. Excitatory Spiking Patterns
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for j in range(3):
    axs[j].eventplot(all_positions_inh_exc[j], colors=[colors[j]] * 4)
    axs[j].set_title(f"{method_names[j]}")
    axs[j].set_xlabel("Time")
    axs[j].set_ylabel("Neuron Index")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Plot 3: Membrane Potential
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for j in range(3):
    axs[j].plot(all_potentials[j], color=colors[j])
    axs[j].set_title(f"{method_names[j]}")
    axs[j].set_xlabel("Time")
    axs[j].set_ylabel("Potential (mV)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Plot 4: Cross-Correlation
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for j in range(3):
    lags, cross_corr_normalized = all_cross_corrs[j]
    axs[j].plot(lags, cross_corr_normalized, color=colors[j])
    axs[j].axvline(0, color="gray", linestyle="--")
    axs[j].set_title(f"{method_names[j]}")
    axs[j].set_xlabel("Lag")
    axs[j].set_ylabel("Normalized Cross-Correlation")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Plot 5: Weight change for inhibitory neurons
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
col = ["red", "blue", "green"]
for j in range(3):
    inhibitory_weights = all_weights[j][
        -N_inh:, :
    ]  # Extract weights for current method
    time_points = np.arange(all_weights[j].shape[1])  # Time axis
    for inh_idx in range(
        inhibitory_weights.shape[0]
    ):  # Iterate over inhibitory neurons
        axs[j].plot(
            time_points, inhibitory_weights[inh_idx, :], alpha=0.7, color=col[j]
        )
    axs[j].set_title(f"{method_names[j]}")
    axs[j].set_xlabel("Time")
    axs[j].set_ylabel("Weight Value")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


def mutual_information(seq1, seq2):
    """
    Calculate the mutual information between two binary sequences.

    Parameters:
        seq1 (list or np.ndarray): First binary sequence.
        seq2 (list or np.ndarray): Second binary sequence.

    Returns:
        float: Mutual information between seq1 and seq2.
    """
    # Ensure the sequences are numpy arrays
    seq1 = np.array(seq1)
    seq2 = np.array(seq2)

    if len(seq1) != len(seq2):
        raise ValueError("Sequences must have the same length.")

    # Joint distribution
    joint_prob, _, _ = np.histogram2d(seq1, seq2, bins=2, range=[[0, 1], [0, 1]])
    joint_prob = joint_prob / joint_prob.sum()

    # Marginal distributions
    p_seq1 = joint_prob.sum(axis=1)
    p_seq2 = joint_prob.sum(axis=0)

    # Mutual information calculation
    mutual_info = 0
    for i in range(2):
        for j in range(2):
            if joint_prob[i, j] > 0:  # Avoid log(0)
                mutual_info += joint_prob[i, j] * np.log(
                    joint_prob[i, j] / (p_seq1[i] * p_seq2[j])
                )

    return mutual_info


l = ["anti_STDP", "anti_Hebb", "STDP"]
for i in range(3):
    print(
        l[i], mutual_information(all_spikes[i][-(N_inh + N_out)], all_spikes[i][-N_out])
    )
