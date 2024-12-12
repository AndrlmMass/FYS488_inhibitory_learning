from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

T = 1000 # ms
lambd = 100 # Hz
N_exc = 8
N_x = 10
V_reset = -80
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
exc_weights = 0.5
inh_weights = -0.5
U_rest = -70
learning_rate = 0.001
V_thresh = -55
plot_data = True
anti_STDP_ = True
anti_hebb_ = False
STDP_= False

weights = np.zeros(shape=(N_x, T))
weights[:8, 0] = exc_weights
weights[8:, 0] = inh_weights

float_data = np.zeros(shape=(N_items, N_x))

# Parameters
duration = 1000        # Total simulation time in ms
dt = 1                 # Time step in ms

# Derived values
time_steps = int(duration / dt)  # Total number of time steps
p_spike = lambd / T       # Spike probability per time step

# Generate binary spike trains
spike_matrix = (np.random.rand(N_x, T) < p_spike).astype(int)
new_stack = np.zeros(time_steps)
spike_matrix = np.vstack([spike_matrix, new_stack])

prev_S = np.zeros(shape=(N_x+1))
prev_S = np.random.randint(low=1000,high=100000, size=N_x+1)
z_trace = np.zeros(shape=(N_x+1, T))

# spike_matrix now contains 1s for spikes and 0s for no spikes
# Extract spike times and neuron indices for plotting
if plot_data:
    positions = [np.where(spike_matrix[n, :] == 1)[0] for n in range(N_x)]
    colors = ["green", "green", "green", "green", "green", "green", "green", "green", "red", "red"]
    plt.eventplot(positions=positions, colors=colors)
    plt.savefig('/home/andreas-massey/Documents/FYS_article/training_spikes.png')
    plt.show()



U = np.full(shape=(T), fill_value=U_rest)

def anti_STDP(prev_S, weights, A_plus, A_minus, tau_plus, tau_minus):
    delta_t = prev_S[-1] - prev_S[:-1]

    for i in range(len(delta_t)):
        if delta_t[i] > max_diff or delta_t[i] < min_diff:
            continue 
        if delta_t[i] > 0:
            delta_w = -A_minus * np.exp(delta_t[i]/tau_minus)
            weights[i] = min(-0.1, delta_w)
        else:
            delta_w = A_plus * np.exp(-delta_t[i]/tau_plus)
            weights[i] = min(-0.1, delta_w)
    return weights

def anti_hebb(z_trace, learning_rate):
    weights = np.multiply(z_trace[-1], z_trace[:-1])*learning_rate
    return weights 

def STDP(prev_S, weights, A_plus, A_minus, tau_plus, tau_minus):
    delta_t = prev_S[-1] - prev_S[:-1]

    for i in range(len(delta_t)):
        if delta_t[i] > max_diff or delta_t[i] > min_diff:
            continue 
        if delta_t[i] > 0:
            delta_w = A_plus * np.exp(-delta_t[i]/tau_plus)
            weights[i] = max(delta_w, 0.01)
        else:
            delta_w = -A_minus * np.exp(delta_t[i]/tau_minus)
            weights[i] = max(delta_w, 0.01)
    return weights

for t in tqdm(range(1, T)):
    # update membrane potential
    I_in = np.dot(weights[:, t-1], spike_matrix[:-1, t-1])
    U_delta = (I_in - ((U[t-1]-U_rest)*(V_thresh-U[t-1]))/(Rm*(V_thresh-U_rest)))/Cm
    U[t] = U[t-1] + U_delta

    # update trace
    z_trace[:, t] += -z_trace[:, t-1]/tau_z 

    # update spike_timing
    if U[t] > V_thresh:
        spike_matrix[-1, t] = 1
        U[t] = V_reset

    prev_S = np.where(spike_matrix[:, t] == 1, 0, prev_S + 1)
    z_trace[:, t][spike_matrix[:, t] == 1] += 1 
    
    # update weights
    if anti_STDP_:
        weights[N_exc:, t] = anti_STDP(prev_S=prev_S[N_exc:], weights=weights[N_exc:, t-1], A_plus=A_plus, A_minus=A_minus, tau_plus=tau_plus, tau_minus=tau_minus)
    elif anti_hebb_:
        weights[:, t] = anti_hebb(z_trace=z_trace, learning_rate=learning_rate)
    elif STDP_:
        weights[:, t] = STDP(prev_S=prev_S[:, t], weights=weights[:, t-1], A_plus=A_plus, A_minus=A_minus, tau_plus=tau_plus, tau_minus=tau_minus)


# plot results
positions = [np.where(spike_matrix[n, :] == 1)[0] for n in range(N_x+1)]
#color = ["green", "green", "green", "green", "green", "green", "green", "green", "red", "red", "blue"]
plt.eventplot(positions=positions, colors="black")
plt.show()

# plot membrane potential
plt.plot(U)
plt.show()

# plot weights over time
for neuron_idx, weights in enumerate(weights):
    plt.plot(weights, label=f"Neuron {neuron_idx+1}")
plt.show()



