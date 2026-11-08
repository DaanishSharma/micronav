"""Rotations (chapter 1).

A rotation can be written two ways:
- a 3x3 matrix R: multiply a vector by it to rotate the vector
- a rotation vector w: its direction is the axis, its length is the angle in radians

`exp_so3` turns a rotation vector into a matrix (Rodrigues' formula), and `log_so3` goes back.
Gyro integration is "add up many tiny rotation vectors".

Naming: R_AB takes a vector written in frame B and gives it in frame A.
So R_WI (IMU -> world) is "the orientation of the IMU in the world".
"""
import numpy as np


def skew(w):
    """Matrix such that skew(a) @ b == cross(a, b)."""
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def exp_so3(w):
    """Rotation vector (axis * angle) -> rotation matrix."""
    w = np.asarray(w, float)
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + skew(w)
    K = skew(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def log_so3(R):
    """Rotation matrix -> rotation vector. Valid for angles below pi."""
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    return v if th < 1e-9 else v * th / np.sin(th)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ypr_to_R(yaw, pitch, roll):
    """Yaw about z, then pitch about y, then roll about x (all radians)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def R_to_ypr(R):
    """Inverse of `ypr_to_R`: returns (yaw, pitch, roll) in radians."""
    return np.arctan2(R[1, 0], R[0, 0]), np.arcsin(-R[2, 0]), np.arctan2(R[2, 1], R[2, 2])


def angle_between(a, b):
    """Angle between two vectors, in degrees."""
    c = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def rotation_angle_deg(R_a, R_b):
    """How far apart two orientations are, in degrees."""
    return np.degrees(np.linalg.norm(log_so3(R_a.T @ R_b)))


def make_T(R, t):
    """4x4 pose from a rotation and a translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
