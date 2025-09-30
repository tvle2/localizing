#CRB bound
import jax.numpy as jnp
from tensorflow import convert_to_tensor,boolean_mask
print(5600/140)
tau_min = 10 ** (-3) * 20 * 5 * 3.14
G = 5
F = 2
N = 8

timeSet = [(2 ** (N - 1 + 1)) * tau_min]
for n in range(1, N + 1):
    n_ts = G + F * (n - 1)
    t_high = (2 ** (N - n + 1)) * tau_min
    t_low = (2 ** (N - n)) * tau_min
    deta_t = ((t_high - t_low) * jnp.arange(n_ts) /
              (n_ts - 1) + t_low)
    deta_t = deta_t[::-1]
    timeSet.extend(deta_t[1:])

# Create a tensor representing histogram outcomes (initially zeros) as float32
# Create a tensor representing histogram outcomes (initially zeros) as float32
data_numpy = [float(x) for x in timeSet]

# Convert to TensorFlow tensor
ctimeSet = convert_to_tensor(data_numpy)
filtered_tensor = boolean_mask(ctimeSet, ctimeSet <= 96.0)

data_numpy1 = [float(x)+240 for x in timeSet]
evolution_time = 3.14/ 10
print(sum(data_numpy1),(22000-(evolution_time+240)*50)/260)