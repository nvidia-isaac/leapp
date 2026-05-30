import gc
import os

import numpy as np
import warp as wp

wp.init()

num_particles = 1_000_000
dt = 0.01
device = "cuda:0"
GRAPH_PATH = os.path.join(os.path.dirname(__file__), "warp_graph")


@wp.kernel
def gravity_step(pos: wp.array[wp.vec3], vel: wp.array[wp.vec3]):
    i = wp.tid()
    position = pos[i]
    dist_sq = wp.length_sq(position) + 0.01  # softened distance
    acc = -1000.0 / dist_sq * wp.normalize(position)  # gravitational pull toward origin
    vel[i] = vel[i] + acc * dt
    pos[i] = pos[i] + vel[i] * dt


def simulate(positions: wp.array, velocities: wp.array) -> None:
    for _ in range(100):
        wp.launch(gravity_step, dim=num_particles, inputs=[positions, velocities], device=device)


rng = np.random.default_rng(42)
positions = wp.array(rng.normal(size=(num_particles, 3)), dtype=wp.vec3, device=device)
velocities = wp.array(rng.normal(size=(num_particles, 3)), dtype=wp.vec3, device=device)

# apic=True is required for capture_save / capture_load (.wrp serialization).
with wp.ScopedCapture(device=device, force_module_load=True, apic=True) as capture:
    simulate(positions, velocities)

graph = capture.graph

# Replay once in-process before persisting.
wp.capture_launch(graph)
wp.synchronize_device()
print("in-process replay:", positions.numpy()[:5])

# Serialize graph, kernel modules, and buffer bindings to disk.
wp.capture_save(
    graph,
    GRAPH_PATH,
    inputs={"positions": positions, "velocities": velocities},
    outputs={"positions": positions, "velocities": velocities},
)

# Drop the original capture and GPU arrays (simulates tearing down the session).
del graph, positions, velocities, capture
gc.collect()

# Reload from disk: kernels come from warp_graph_modules/, not from @wp.kernel above.
loaded = wp.capture_load(GRAPH_PATH, device=device)
wp.capture_launch(loaded)
wp.synchronize_device()

out_positions = wp.empty(num_particles, dtype=wp.vec3, device=device)
loaded.get_param("positions", out_positions)
print("loaded replay:", out_positions.numpy()[:5])
