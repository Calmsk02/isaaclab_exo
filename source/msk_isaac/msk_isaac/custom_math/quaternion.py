import torch
from typing import Tuple

###
# Quaternion calculation (Isaac Lab convention: [w, x, y, z])
###

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    q: (N, 4) in [w, x, y, z]
    returns conjugate(q)
    """
    qc = q.clone()
    qc[:, 1:] = -qc[:, 1:]
    return qc


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product for quaternions in [w, x, y, z]
    q = q1 * q2
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector v by quaternion q
    q: (N, 4) [w, x, y, z]
    v: (N, 3)
    return: (N, 3)
    """
    zeros = torch.zeros((v.shape[0], 1), device=v.device, dtype=v.dtype)
    v_quat = torch.cat([zeros, v], dim=-1)   # [0, vx, vy, vz]

    return quat_mul(quat_mul(q, v_quat), quat_conjugate(q))[:, 1:]


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Inverse rotate vector v by quaternion q
    v_rot = conj(q) * [0, v] * q
    """
    zeros = torch.zeros((v.shape[0], 1), device=v.device, dtype=v.dtype)
    v_quat = torch.cat([zeros, v], dim=-1)   # [0, vx, vy, vz]

    return quat_mul(quat_mul(quat_conjugate(q), v_quat), q)[:, 1:]


def quat_error_as_rotvec(q_current: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    """
    Quaternion error as rotation vector
    q_err = q_target * conj(q_current)

    inputs: [w, x, y, z]
    output: rotation vector (N, 3)
    """
    q_err = quat_mul(q_target, quat_conjugate(q_current))

    q_w = q_err[:, 0:1]
    q_xyz = q_err[:, 1:]

    # shortest path
    sign = torch.sign(q_w)
    sign[sign == 0] = 1.0
    q_w = q_w * sign
    q_xyz = q_xyz * sign

    norm_xyz = torch.norm(q_xyz, dim=-1, keepdim=True).clamp_min(1e-8)
    angle = 2.0 * torch.atan2(norm_xyz, q_w.clamp_min(1e-8))
    axis = q_xyz / norm_xyz

    rotvec = axis * angle
    return rotvec


def quat_to_euler(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert quaternion [w, x, y, z] to roll, pitch, yaw
    returns:
        roll  - rotation around x
        pitch - rotation around y
        yaw   - rotation around z
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)

    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw