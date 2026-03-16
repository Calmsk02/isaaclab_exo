import torch
import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg, OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

# Basic IK Solver Class
class BaseIKSolver():
    def __init__(self,
                 name: str,
                 sim: sim_utils.SimulationContext, 
                 scene: InteractiveScene,
                 robot_entity_cfg: SceneEntityCfg,
                 ):
        
        # Extract scene entities
        self.robot = scene["robot"]
        self.robot_entity_cfg = robot_entity_cfg
        self.num_envs = scene.num_envs
        self.device = sim.device
        self.env_origins = scene.env_origins
        self.sim_dt = sim.get_physics_dt()

        # Get Joint Indexes
        self.joint_ids = robot_entity_cfg.joint_ids
        self.body_ids = robot_entity_cfg.body_ids # [Base link, EE link]

        # Get Base/EE Link Indexes
        self.base_link_idx = self.body_ids[0]
        self.ee_link_idx = self.body_ids[1]
        
        # Get End-Effector Index
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.ee_link_idx - 1
        else:
            self.ee_jacobi_idx = self.ee_link_idx

        # Set Frame Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path=f"/Visuals/{name}_ee_current")
        )
        self.goal_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path=f"/Visuals/{name}_ee_goal")
        )

        print("[", name, "_IK_Solver]", "Joints:", self.joint_ids, 
              " Base:", robot_entity_cfg.body_names[0], "(", self.base_link_idx, ")", 
              " EE:", robot_entity_cfg.body_names[1], "(", self.ee_link_idx, ")"
              )
        
        (   
            _,
            self.init_ee_pose_b,
            _,
            self.init_ee_pose_w,
            _,
            _
        ) = self._update_states()

    """ 
    return: 
    joint_pos, joint_vel,
    ee_pose_w, ee_pose_b, base_pose_w,
    ee_jacobian_w, ee_jacobian_b
    """
    def _update_states(self):
        # Jacobian w.r.t. world frame
        # Tensor is 4 dimentions: (num_envs, num_links(except fixed base link), spatial velocity(6), num_joints)
        ee_jacobian_w = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        
        # End effector / base pose w.r.t. world frame
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_link_idx]
        base_pose_w = self.robot.data.body_pose_w[:, self.base_link_idx]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            base_pose_w[:, 0:3], base_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)

        # Jacobian w.r.t. base frame
        ee_jacobian_b = ee_jacobian_w.clone()
        base_rot_matrix = matrix_from_quat(quat_inv(base_pose_w[:, 3:7]))
        ee_jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, ee_jacobian_b[:, :3, :])
        ee_jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, ee_jacobian_b[:, 3:, :])
        
        # Current joint positions
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        # Current joint velocities
        joint_vel = self.robot.data.joint_vel[:, self.robot_entity_cfg.joint_ids]

        return ( 
            ee_jacobian_b,
            ee_pose_b,
            base_pose_w,
            ee_pose_w,
            joint_pos,
            joint_vel
        )
    
    def _update_marker(self, ee_pose_w, base_pose_w, ee_target_pose_b):
        # Update marker
        # change target command w.r.t. world frame
        ee_target_pos_w, ee_target_quat_w = combine_frame_transforms(
            base_pose_w[:, 0:3], base_pose_w[:, 3:7], ee_target_pose_b[:, 0:3], ee_target_pose_b[:, 3:7]
        )
        # Update marker
        # change target command w.r.t. world frame
        self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        self.goal_marker.visualize(ee_target_pos_w, ee_target_quat_w)


# Differential Inverse Kinematics Solver
class DiffIKSolver(BaseIKSolver):
    def __init__(self,
                 name: str,
                 sim: sim_utils.SimulationContext, 
                 scene: InteractiveScene,
                 robot_entity_cfg: SceneEntityCfg,
                 command_type="pose",
                 use_relative_mode=False,
                 ik_method="dls"):
        
        super().__init__(name, sim, scene, robot_entity_cfg)
        
        # class params
        self.command_type = command_type

        # Create controller
        # action_dim = 7 (x,y,z,qx,qy,qz,qw) for command type "pose"
        self.diff_ik_cfg = DifferentialIKControllerCfg(
            command_type=command_type, 
            use_relative_mode=use_relative_mode, 
            ik_method=ik_method
        )
        self.diff_ik_controller = DifferentialIKController(
            self.diff_ik_cfg, 
            num_envs=self.num_envs, 
            device=self.device
        )
        
        self.reset()
        self.set_command(self.init_ee_pose_b)

    def reset(self):
        self.diff_ik_controller.reset()
        self.ee_marker.visualize(self.init_ee_pose_w[:, 0:3], self.init_ee_pose_w[:, 3:7])
        self.goal_marker.visualize(self.init_ee_pose_w[:, 0:3], self.init_ee_pose_w[:, 3:7])
    
    def set_command(self, commands: torch.Tensor):
        (
            _,
            ee_pose_b,
            _,
            _,
            _,
            _
        ) = self._update_states()

        self.commands = commands.clone()
        if self.command_type == "position":
            self.diff_ik_controller.set_command(self.commands[:, 0:3], ee_pos=ee_pose_b[:, 0:3], ee_quat=ee_pose_b[:, 3:7])
        else:
            self.diff_ik_controller.set_command(self.commands)

    def compute(self):
        (
            ee_jacobian_b,
            ee_pose_b,
            base_pose_w,
            ee_pose_w,
            joint_pos,
            _
        ) = self._update_states()

        # Compute desired joint positions
        joint_pos_des =self.diff_ik_controller.compute(ee_pose_b[:,0:3], ee_pose_b[:,3:7], ee_jacobian_b, joint_pos)

        # Update marker
        ee_target_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        ee_target_pose_b[:] = self.commands[:, :7]
        self._update_marker(ee_pose_w, base_pose_w, ee_target_pose_b)

        return joint_pos_des


# Operational Space Control Inverse Kinematics Solver
class OSCIKSolver(BaseIKSolver):
    def __init__(self,
                 name: str,
                 sim: sim_utils.SimulationContext, 
                 scene: InteractiveScene,
                 robot_entity_cfg: SceneEntityCfg,
                 target_types=["pose_abs"], # pose (x,y,z,qx,qy,qz,qw), wrench (fx,fy,fz,mx,my,mz)
                 impedance_mode="variable_kp",
                 inertial_dynamics_decoupling=True,
                 partial_inertial_dynamics_decoupling=False,
                 gravity_compensation=True,
                 motion_damping_ratio_task=1.0,
                 contact_wrench_stiffness_task=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 motion_control_axes_task=[1, 1, 1, 1, 1, 1],
                 contact_wrench_control_axes_task=[0, 0, 0, 0, 0, 0],
                 nullspace_control="none",
                 contact_forces=None):
        
        super().__init__(name, sim, scene, robot_entity_cfg)
        self.contact_forces = None
        if contact_forces is not None:
            self.contact_forces = scene[contact_forces]
        
        self.osc_cfg = OperationalSpaceControllerCfg(
            target_types=target_types,
            impedance_mode=impedance_mode,
            inertial_dynamics_decoupling=inertial_dynamics_decoupling,
            partial_inertial_dynamics_decoupling=partial_inertial_dynamics_decoupling,
            gravity_compensation=gravity_compensation,
            motion_damping_ratio_task=motion_damping_ratio_task,
            contact_wrench_stiffness_task=contact_wrench_stiffness_task,
            motion_control_axes_task=motion_control_axes_task,
            contact_wrench_control_axes_task=contact_wrench_control_axes_task,
            nullspace_control=nullspace_control
        )
        self.osc = OperationalSpaceController(
            self.osc_cfg, 
            num_envs=self.num_envs, 
            device=self.device
        )
        
        self.reset()
        self.set_command(self.init_ee_pose_b)

    def reset(self):
        self.osc.reset()
        self.ee_marker.visualize(self.init_ee_pose_w[:, 0:3], self.init_ee_pose_w[:, 3:7])
        self.goal_marker.visualize(self.init_ee_pose_w[:, 0:3], self.init_ee_pose_w[:, 3:7])

    def update_states(self):
        (
            ee_jacobian_b,
            ee_pose_b,
            base_pose_w,
            ee_pose_w,
            joint_pos,
            joint_vel,
        ) = self._update_states()
        
        # dynamics properties
        mass_matrix = self.robot.root_physx_view.get_generalized_mass_matrices()[:, self.joint_ids, :][:, :, self.joint_ids]
        gravity = self.robot.root_physx_view.get_gravity_compensation_forces()[:, self.joint_ids]

        # compute current velocity of EE
        ee_vel_w = self.robot.data.body_vel_w[:, self.ee_link_idx, :]
        base_vel_w = self.robot.data.body_vel_w[:, self.base_link_idx, :]
        relative_vel_w = ee_vel_w - base_vel_w
        ee_lin_vel_b = quat_apply_inverse(base_pose_w[:, 3:7], relative_vel_w[:, 0:3])
        ee_ang_vel_b = quat_apply_inverse(base_pose_w[:, 3:7], relative_vel_w[:, 3:7])
        ee_vel_b = torch.cat([ee_lin_vel_b, ee_ang_vel_b], dim=-1)

        ee_force_b = torch.zeros(self.num_envs, 6, device=self.device)
        if self.contact_forces is not None:
            sim_dt = self.sim_dt
            self.contact_forces.update(sim_dt)
            ee_force_b, _ = torch.max(torch.mean(self.contact_forces.data.net_forces_w_history, dim=1), dim=1)

        return (
            ee_jacobian_b,
            mass_matrix,
            gravity,
            ee_pose_b,
            ee_vel_b,
            base_pose_w,
            ee_pose_w,
            ee_force_b,
            joint_pos,
            joint_vel
        )
    
    def set_command(self, commands: torch.Tensor):
        (
            _,
            ee_pose_b,
            _,
            _,
            _,
            _,
        ) = self._update_states()

        # 원래 목표는 marker용으로 보존
        self.commands = commands.clone()

        # OSC에 넣을 내부 command는 별도 복사본 사용
        osc_command = commands.clone()

        # change command w.r.t. task frame
        ee_target_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        ee_task_frame_pose_b = ee_target_pose_b.clone()

        for target_type in self.osc.cfg.target_types:
            if target_type == "pose_abs":
                ee_target_pose_b[:] = self.commands[:, :7]
                ee_task_frame_pose_b = ee_target_pose_b.clone()
                osc_command[:, 0:3], osc_command[:, 3:7] = subtract_frame_transforms(
                    ee_task_frame_pose_b[:, 0:3],
                    ee_task_frame_pose_b[:, 3:7],
                    osc_command[:, 0:3],
                    osc_command[:, 3:7],
                )
            elif target_type == "wrench_abs":
                pass
            else:
                raise ValueError("Undefined target_type within set_command().")

        self.osc.set_command(
            command=osc_command,
            current_ee_pose_b=ee_pose_b,
            current_task_frame_pose_b=ee_task_frame_pose_b,
        )

    def compute(self):
        (
            ee_jacobian_b,
            mass_matrix,
            gravity,
            ee_pose_b,
            ee_vel_b,
            base_pose_w,
            ee_pose_w,
            ee_force_b,
            joint_pos,
            joint_vel
        ) = self.update_states()

        joint_centers = torch.mean(self.robot.data.soft_joint_pos_limits[:, self.joint_ids, :], dim=-1)

        joint_efforts = self.osc.compute(
            jacobian_b=ee_jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            current_ee_force_b=ee_force_b,
            mass_matrix=mass_matrix,
            gravity=gravity,
            current_joint_pos=joint_pos,
            current_joint_vel=joint_vel,
            nullspace_joint_pos_target=joint_centers,
        )

        # Update marker
        ee_target_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        ee_target_pose_b[:] = self.commands[:, :7]
        self._update_marker(ee_pose_w, base_pose_w, ee_target_pose_b)

        return joint_efforts