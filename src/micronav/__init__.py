"""micronav: sensor fusion and navigation from scratch, in plain numpy."""
from .geo import LocalFrame
from .imu import GRAVITY, dead_reckon, gyro_bias_from_still, integrate_gyro
from .io import Recording, load_phyphox
from .sim import Sim, simulate_gps, simulate_walk
from .so3 import (R_to_ypr, angle_between, exp_so3, log_so3, make_T, rot_x, rot_y,
                  rot_z, rotation_angle_deg, skew, ypr_to_R)

__version__ = "0.0.1"
