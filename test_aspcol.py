import numpy as np
import aspcol.distance as dt


vec1 = np.array([1, 1])
vec2 = np.array([1, 2])


angle = dt.angular_distance(vec1, vec2, sign_invariant=False)
print(f'angle: {angle}')