import torch
import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_inv,
    subtract_frame_transforms,
)

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

class WholeBodyDIKSolver:
    def __init__(
        self,
        name: str,
        sim: sim_utils.SimulationContext,
        scene: InteractiveScene,
        whole_body_joint_ids: torch.Tensor,
        task_body_names: dict,
        base_body_name: str = "base_link",
        position_only_tasks: dict | None = None,
        task_weights: dict | None = None,
        damping: float = 0.05,
        pos_gain: float = 1.0,
        rot_gain: float = 0.5,
    ):
        self.name = name
        self.robot = scene["robot"]
        self.scene = scene
        self.device = sim.device
        self.num_envs = scene.num_envs
        self.sim_dt = sim.get_physics_dt()

        self.whole_body_joint_ids = whole_body_joint_ids
        self.task_body_names = task_body_names
        self.position_only_tasks = position_only_tasks or {}
        self.task_weights = task_weights or {}
        self.damping = damping
        self.pos_gain = pos_gain
        self.rot_gain = rot_gain

        # root/base
        base_ids, _ = self.robot.find_bodies([base_body_name])
        self.base_body_idx = int(base_ids[0])

        # task body indices
        self.task_body_indices = {}
        self.task_jacobi_indices = {}
        for task_name, body_name in self.task_body_names.items():
            body_ids, _ = self.robot.find_bodies([body_name])
            body_idx = int(body_ids[0])
            self.task_body_indices[task_name] = body_idx

            if self.robot.is_fixed_base:
                self.task_jacobi_indices[task_name] = body_idx - 1
            else:
                self.task_jacobi_indices[task_name] = body_idx

        # initial targets
        self.target_poses_b = {}
        self._capture_initial_targets()

        # markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
        self.current_markers = {}
        self.goal_markers = {}
        for task_name in self.task_body_names.keys():
            self.current_markers[task_name] = VisualizationMarkers(
                frame_marker_cfg.replace(prim_path=f"/Visuals/{name}_{task_name}_current")
            )
            self.goal_markers[task_name] = VisualizationMarkers(
                frame_marker_cfg.replace(prim_path=f"/Visuals/{name}_{task_name}_goal")
            )

    def _capture_initial_targets(self):
        body_pose_w = self.robot.data.body_pose_w
        base_pose_w = body_pose_w[:, self.base_body_idx]

        for task_name, body_idx in self.task_body_indices.items():
            ee_pose_w = body_pose_w[:, body_idx]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                base_pose_w[:, 0:3], base_pose_w[:, 3:7],
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
            )
            self.target_poses_b[task_name] = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    def reset(self):
        self._capture_initial_targets()

    def set_task_command(self, task_name: str, target_pose_b: torch.Tensor):
        self.target_poses_b[task_name] = target_pose_b.clone()

    def get_task_pose_b(self, task_name: str) -> torch.Tensor:
        body_pose_w = self.robot.data.body_pose_w
        base_pose_w = body_pose_w[:, self.base_body_idx]
        ee_pose_w = body_pose_w[:, self.task_body_indices[task_name]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            base_pose_w[:, 0:3], base_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
        )
        return torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    def _get_base_rot_matrix(self) -> torch.Tensor:
        base_pose_w = self.robot.data.body_pose_w[:, self.base_body_idx]
        return matrix_from_quat(quat_inv(base_pose_w[:, 3:7]))

    def _get_task_jacobian_b(self, task_name: str) -> torch.Tensor:
        jac_w = self.robot.root_physx_view.get_jacobians()[
            :, self.task_jacobi_indices[task_name], :, self.whole_body_joint_ids
        ]
        R_bw = self._get_base_rot_matrix()
        jac_b = jac_w.clone()
        jac_b[:, :3, :] = torch.bmm(R_bw, jac_b[:, :3, :])
        jac_b[:, 3:, :] = torch.bmm(R_bw, jac_b[:, 3:, :])
        return jac_b

    def _build_stacked_system(self):
        J_blocks = []
        e_blocks = []

        body_pose_w = self.robot.data.body_pose_w
        base_pose_w = body_pose_w[:, self.base_body_idx]

        for task_name, body_idx in self.task_body_indices.items():
            target_pose_b = self.target_poses_b[task_name]
            current_pose_w = body_pose_w[:, body_idx]

            current_pos_b, current_quat_b = subtract_frame_transforms(
                base_pose_w[:, 0:3], base_pose_w[:, 3:7],
                current_pose_w[:, 0:3], current_pose_w[:, 3:7],
            )

            current_pose_b = torch.cat([current_pos_b, current_quat_b], dim=-1)
            jac_b = self._get_task_jacobian_b(task_name)

            pos_err = (target_pose_b[:, 0:3] - current_pose_b[:, 0:3]) * self.pos_gain

            position_only = self.position_only_tasks.get(task_name, False)
            weight = self.task_weights.get(task_name, 1.0)

            if position_only:
                J_task = jac_b[:, :3, :]
                e_task = pos_err
            else:
                rot_err = quat_error_as_rotvec(current_pose_b[:, 3:7], target_pose_b[:, 3:7]) * self.rot_gain
                J_task = jac_b
                e_task = torch.cat([pos_err, rot_err], dim=-1)

            J_task = J_task * weight
            e_task = e_task * weight

            J_blocks.append(J_task)
            e_blocks.append(e_task)

        J_stack = torch.cat(J_blocks, dim=1)   # (N, task_dim_total, n_joints)
        e_stack = torch.cat(e_blocks, dim=1)   # (N, task_dim_total)
        return J_stack, e_stack

    def compute(self):
        joint_pos = self.robot.data.joint_pos[:, self.whole_body_joint_ids]

        J, e = self._build_stacked_system()
        n_envs, task_dim, n_joints = J.shape

        I_task = torch.eye(task_dim, device=self.device).unsqueeze(0).repeat(n_envs, 1, 1)
        JJt = torch.bmm(J, J.transpose(1, 2))
        damping_term = (self.damping ** 2) * I_task

        dq = torch.bmm(
            J.transpose(1, 2),
            torch.linalg.solve(JJt + damping_term, e.unsqueeze(-1))
        ).squeeze(-1)

        joint_pos_des = joint_pos + dq

        self._update_markers()
        return joint_pos_des

    def _update_markers(self):
        body_pose_w = self.robot.data.body_pose_w
        base_pose_w = body_pose_w[:, self.base_body_idx]

        for task_name, body_idx in self.task_body_indices.items():
            current_pose_w = body_pose_w[:, body_idx]
            target_pose_b = self.target_poses_b[task_name]

            target_pos_w, target_quat_w = combine_frame_transforms(
                base_pose_w[:, 0:3], base_pose_w[:, 3:7],
                target_pose_b[:, 0:3], target_pose_b[:, 3:7],
            )

            self.current_markers[task_name].visualize(
                current_pose_w[:, 0:3], current_pose_w[:, 3:7]
            )
            self.goal_markers[task_name].visualize(
                target_pos_w, target_quat_w
            )