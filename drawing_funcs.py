import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

###########################
### Symmetrical Hebbian ###
###########################

# Define x-axis range
x_axis = np.arange(-40, 40, 0.001)

# Mean and standard deviation
mean = 0
sd = 10

# Compute the normal PDF
y = norm.pdf(x_axis, mean, sd)

# Ensure the curve flattens out towards zero by taking the absolute value
y_positive = np.abs(y)

# Plot the adjusted distribution
plt.plot(x_axis, y_positive, label="Flattened Normal Distribution", color="green")
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Horizontal line at 0
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)  # Vertical line at 0
plt.fill_between(x_axis, y, 0, where=(y > 0), color="green", alpha=0.5)
plt.ylabel(r"$\Delta w$", fontsize=16)
plt.xlabel(r"$\Delta t$", fontsize=16)
plt.show()

##############################
### Symmetric Anti-Hebbian ###
##############################

# Define x-axis range
x_axis = np.arange(-40, 40, 0.001)

# Mean and standard deviation
mean = 0
sd = 10

# Compute the anti-Hebbian rule: Flip the Hebbian curve by multiplying by -1
y = -norm.pdf(x_axis, mean, sd)

# Plot the anti-Hebbian rule
plt.figure(figsize=(8, 6))
plt.plot(x_axis, y, label="Anti-Hebbian Rule", color="red")

# Fill the negative area under the curve
plt.fill_between(x_axis, y, 0, where=(y < 0), color="red", alpha=0.5, label="Negative Weight Change")

# Add labels, title, and legend
plt.ylabel(r"$\Delta w$", fontsize=16)
plt.xlabel(r"$\Delta t$", fontsize=16)
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)  # Horizontal line at 0
plt.axvline(0, color="black", linestyle="--", linewidth=0.8)  # Vertical line at 0

# Show the plot
plt.show()


##########################
### STDP rule plotting ###
##########################


# Parameters for STDP
A_plus = 1.0     # Scaling factor for LTP
A_minus = -1.0   # Scaling factor for LTD
tau_plus = 20.0  # Time constant for LTP
tau_minus = 20.0 # Time constant for LTD

# Time differences (Δt)
delta_t = np.linspace(-50, 50, 1000)  # From -50ms to 50ms

# Flipped STDP Rule
delta_w = np.piecewise(delta_t, 
                       [delta_t > 0, delta_t <= 0], 
                       [lambda t: A_minus * np.exp(-t / tau_plus),  # Decrease when Δt > 0
                        lambda t: A_plus * np.exp(t / tau_minus)])  # Increase when Δt < 0

# Plotting the flipped STDP curve
plt.figure(figsize=(8, 6))

# Positive segment (Δt < 0, weight increase)
positive_mask = delta_w > 0
plt.plot(delta_t[positive_mask], delta_w[positive_mask], color="green")

# Negative segment (Δt > 0, weight decrease)
negative_mask = delta_w < 0
plt.plot(delta_t[negative_mask], delta_w[negative_mask], color="red")

# Fill positive and negative areas
plt.fill_between(delta_t, delta_w, 0, where=(delta_w > 0), color="green", alpha=0.3)
plt.fill_between(delta_t, delta_w, 0, where=(delta_w < 0), color="red", alpha=0.3)

# Add horizontal and vertical lines for reference
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Horizontal line at 0
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)  # Vertical line at 0

# Add labels, title, and legend
plt.xlabel(r"$\Delta t$ (ms)", fontsize=16)
plt.ylabel(r"$\Delta w$", fontsize=16)

# Show the plot
plt.show()


###############################
### Anti-STDP rule plotting ###
###############################


# Parameters for STDP
A_plus = 1.0     # Scaling factor for LTP
A_minus = -1.0   # Scaling factor for LTD
tau_plus = 20.0  # Time constant for LTP
tau_minus = 20.0 # Time constant for LTD

# Time differences (Δt)
delta_t = np.linspace(-50, 50, 1000)  # From -50ms to 50ms

# STDP Rule
delta_w = np.piecewise(delta_t, 
                       [delta_t > 0, delta_t <= 0], 
                       [lambda t: A_plus * np.exp(-t / tau_plus), 
                        lambda t: A_minus * np.exp(t / tau_minus)])

# Plotting the STDP curve
plt.figure(figsize=(8, 6))

# Positive segment
positive_mask = delta_w > 0
plt.plot(delta_t[positive_mask], delta_w[positive_mask], color="green")

# Negative segment
negative_mask = delta_w < 0
plt.plot(delta_t[negative_mask], delta_w[negative_mask], color="red")

# Fill positive and negative areas
plt.fill_between(delta_t, delta_w, 0, where=(delta_w > 0), color="green", alpha=0.3)
plt.fill_between(delta_t, delta_w, 0, where=(delta_w < 0), color="red", alpha=0.3)

# Add horizontal and vertical lines for reference
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Horizontal line at 0
plt.axvline(0, color='black', linestyle='--', linewidth=0.8)  # Vertical line at 0

# Add labels, title, and legend
plt.xlabel(r"$\Delta t$ (ms)", fontsize=16)
plt.ylabel(r"$\Delta w$", fontsize=16)

# Show the plot
plt.show()



