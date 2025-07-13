import pathwaysutils

print("***** initializing pathwaysutils")
pathwaysutils.initialize()
print("***** pathwaysutils initialized")

import numpy as np
import jax
from jax.experimental import colocated_python

print("***** jax version on cpu host:", jax.__version__)

print("***** getting tpu devices")
tpu_devices = jax.devices()
print("***** getting cpu devices")
cpu_devices = colocated_python.colocated_cpu_devices(tpu_devices)
print("***** cpu devices: ", cpu_devices)

import cloudpickle

print("***** JAX_PLATFORMS is 'proxy'. Setting up pathways colocated python checkpointing.")
print(f"***** Using jax version {jax.__version__} and cloudpickle version {cloudpickle.__version__}")


print("***** def add_one")


@colocated_python.colocated_python
def add_one(x):
  print("***** jax version:", jax.__version__)
  print("***** add_one")
  return x+1


print("***** creating input")
x = np.array(1)
print("***** putting on device")
x = jax.device_put(x, cpu_devices[0])

print("***** adding one")
out = add_one(x)  # <----- Failure
print("***** getting out")
out = jax.device_get(out)
print("***** out: ", out)
