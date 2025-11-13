import numpy as np

c = np.array([0.2, -0.2, 0.2])
lookat = np.array([0, 0, 0.])
forward = lookat - c
forward = forward / np.linalg.norm(forward)  # [-0.577, -0.577, -0.577]

# Choose world up vector (approximate global up)
world_up = np.array([0, 0, 1])

# Right vector = forward x up
right = np.cross(forward, world_up)
right = right / np.linalg.norm(right)  # Normalize

# Recompute up = right x forward to ensure orthogonality
up = np.cross(right, forward)
up = up / np.linalg.norm(up)

print("x right: ", right)
print("y up: ", up)
print("z backward: ", -forward)
