"""Helper functions for X, Y, Z rotations for use in bqskit.ir.gates."""
from __future__ import annotations

import numpy as np

def rot_x(theta: float):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],[-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)

def rot_x_jac(theta: float):
    return np.array([[-1/2*np.sin(theta/2), -1j/2*np.cos(theta/2)], [-1j/2*np.cos(theta/2), -1/2*np.sin(theta/2)]], dtype=np.complex128)

def rot_y(theta: float):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],[np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)

def rot_y_jac(theta: float):
    return np.array([[-1/2*np.sin(theta/2), -1/2*np.cos(theta/2)], [1/2*np.cos(theta/2), -1/2*np.sin(theta/2)]], dtype=np.complex128)

def rot_z(theta: float):
    return np.array([[np.exp(-1j*theta/2), 0],[0, np.exp(1j*theta/2)]], dtype=np.complex128)

def rot_z_jac(theta: float):
    return np.array([[-1j/2*np.exp(-1j*theta/2), 0], [0, 1j/2*np.exp(1j*theta/2)]], dtype=np.complex128)
