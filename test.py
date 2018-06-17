from quaternions import Quaternion
import numpy as np

x_axis_unit = (1, 0, 0)
y_axis_unit = (0, 1, 0)
z_axis_unit = (0, 0, 1)

r1 = Quaternion.from_axisangle(np.pi / 2, x_axis_unit)
r2 = Quaternion.from_axisangle(np.pi / 2, y_axis_unit)
r3 = Quaternion.from_axisangle(np.pi / 2, z_axis_unit)

# Quaternion - vector multiplication
v = r1 * y_axis_unit
v = r2 * v
v = r3 * v

print(v)

# Quaternion - quaternion multiplication
r_total = r3 * r2 * r1
v = r_total * y_axis_unit

print(v)