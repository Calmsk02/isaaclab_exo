import torch
from typing import Tuple

###
# Quaternion calculation
###

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    qc = q.clone()
    qc[:, :3] = -qc[:, :3]
    return qc


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # q: (N, 4) [x, y, z, w]
    # v: (N, 3)
    zeros = torch.zeros((v.shape[0], 1), device=v.device, dtype=v.dtype)
    v_quat = torch.cat([v, zeros], dim=-1) # (N, 4)

    return quat_mul(quat_mul(q, v_quat), quat_conjugate(q))[:, :3]


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # v_rot = conj(q) * [v, 0] * q
    zeros = torch.zeros((v.shape[0], 1), device=v.device, dtype=v.dtype)
    v_quat = torch.cat([v, zeros], dim=-1)

    return quat_mul(quat_mul(quat_conjugate(q), v_quat), q)[:, :3]


def quat_error_as_rotvec(q_current: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    # q_err = q_target * conj(q_current)
    q_err = quat_mul(q_target, quat_conjugate(q_current))
    q_xyz = q_err[:, :3]
    q_w = q_err[:, 3:4]

    # shortest path
    sign = torch.sign(q_w)
    sign[sign == 0] = 1.0
    q_xyz = q_xyz * sign
    q_w = q_w * sign

    norm_xyz = torch.norm(q_xyz, dim=-1, keepdim=True).clamp_min(1e-8)
    angle = 2.0 * torch.atan2(norm_xyz, q_w.clamp_min(1e-8))
    axis = q_xyz / norm_xyz
    rotvec = axis * angle
    return rotvec

def quat_to_euler(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    x,y,z,w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

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
