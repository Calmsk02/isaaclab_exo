"""
Docstring for robots.humanbody
"""
# Import isaaclab packages
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG

# Get asset path
from msk_isaac import ASSET_PATH

# 
from dataclasses import dataclass
import math

# Custom packages
from msk_isaac.custom_math.quaternion import quat_conjugate, quat_mul, quat_rotate, quat_rotate_inverse, quat_to_euler
from msk_isaac.custom_math.math import normalize, normalize_angle

#######################
# Robot Configuration #
#######################
HUMANBODY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/isaac_humanbody_description/urdf/isaac_humanbody/isaac_humanbody.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        rot=(math.cos(math.pi/4), 0.0, 0.0, math.sin(math.pi/4)),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "waist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["waist_pitch"],
            effort_limit_sim=250.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=5.0,
        ),
        "waist_roll": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=5.0,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw"],
            effort_limit_sim=60.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=5.0,
        ),
        "shoulder_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch"],
            effort_limit_sim=40.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=2.0,
        ),
        "shoulder_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_roll"],
            effort_limit_sim=35.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=2.0,
        ),
        "shoulder_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_yaw"],
            effort_limit_sim=30.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=2.0,
        ),
        "elbow_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_elbow_pitch"],
            effort_limit_sim=45.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=2.0,
        ),
        "thigh_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_pitch"],
            effort_limit_sim=150.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=6.0,
        ),
        "thigh_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_roll"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=6.0,
        ),
        "thigh_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_yaw"],
            effort_limit_sim=70.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=6.0,
        ),
        "knee_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_pitch"],
            effort_limit_sim=150.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=4.0,
        ),
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=3.0,
        ),
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll"],
            effort_limit_sim=60.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=3.0,
        ),
    },
)

##################################
# Environment Configuration (RL) #
##################################
@configclass
class HumanbodyEnvCfg(DirectRLEnvCfg):
    # required
    decimation = 1
    episode_length_s = 10.0
    action_space = 23
    observation_space = 7 + 3 * action_space
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=0.01, render_interval=2)
    
    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=500, env_spacing=15.0, replicate_physics=False, clone_in_fabric=False
    )

    # robot
    robot: ArticulationCfg = HUMANBODY_CFG.replace(prim_path="/World/envs/env_.*/Robot")


####################
# Environment (RL) #
####################
class HumanbodyEnv(DirectRLEnv):
    cfg: HumanbodyEnvCfg

    def __init__(self, cfg: HumanbodyEnvCfg, render_mode: str | None=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Torso (base) Frame
        self.init_torso_pos_w = self.robot.data.root_pos_w
        self.init_torso_quat_w = self.robot.data.root_quat_w
        self.inv_start_rot = quat_conjugate(self.init_torso_quat_w)

        # Joint properties
        self._joint_dof_idx, self.joint_names = self.robot.find_joints(".*")
        self.joint_limits = self.robot.data.joint_pos_limits
        self.joint_limits_lower = self.joint_limits[:, :, 0]
        self.joint_limits_upper = self.joint_limits[:, :, 1]
        self.effort_limits = self.robot.data.joint_effort_limits
        self.motor_effort_ratio = torch.ones_like(self.effort_limits, device=self.sim.device)
        _joint_ids, _ = self.robot.find_joints([".*_shoulder_.*", ".*_elbow_.*"])
        self.UB_joint_ids = torch.tensor(_joint_ids, dtype=torch.long, device=self.device)

        # Body
        self._body_idx, self.body_names = self.robot.find_bodies(".*") 

        # Body indexes
        name_to_idx = {name: i for i, name in enumerate(self.robot.body_names)}
        self.torso_idx = name_to_idx["base_link"]
        self.up_body_idx = name_to_idx["upperbody"]
        self.l_foot_idx = name_to_idx["left_foot"]
        self.r_foot_idx = name_to_idx["right_foot"]
        self.l_arm_idx = name_to_idx["left_lowerarm"]
        self.r_arm_idx = name_to_idx["right_lowerarm"]
        
        # Target for forward motion
        self.targets_w = torch.tensor([12, 0, 0],dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.targets_w += self.scene.env_origins
        self.targets_quat_w = torch.tensor([1,0,0,0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        
        # Direction vectors in torso frame
        self.basis_heading_vec = torch.tensor([0, -1, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.basis_up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.basis_side_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        
        # Time step control
        self.time_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim.device)
        
        ### Gait control ###
        target_vel = 1.0 # m/s
        rl_dt = self.cfg.decimation*self.cfg.sim.dt # rl step duration
        # step frequency [gait steps/s]
        f_min = 1.4
        f_max = 2.0
        v_ref = 1.0
        speed_ratio = max(min(target_vel / v_ref, 1.0), 0.0)
        f_steps = f_min + (f_max -f_min) * speed_ratio
        # step length [m]
        step_length = target_vel / f_steps
        self.foot_diff_amplitude = 0.5 * step_length
        # gait period [time_step/gait steps]
        self.period = max(10, int(round(1.0 / (f_steps * rl_dt))))
        
        # COM calculation
        self.mass = self.robot.root_physx_view.get_masses().to(self.sim.device)

        # Other params
        self.termination_height: float = 0.8
        self.joint_vel_scale: float = 0.1

        self.print_robot_info()
        self.setup_visual_markers()
        
    def setup_visual_markers(self):
        # Markers for visualization
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.15, 0.15, 0.15)
        self.base_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path=f"/Visuals/base_marker")
        )
        self.goal_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path=f"/Visuals/goal_marker")
        )
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.up_body_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path=f"/Visuals/upbody_marker")
        )
        self.l_arm_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path=f"/Visuals/larm_marker")
        )
        self.r_arm_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path=f"/Visuals/rarm_marker")
        ) 
        self.l_foot_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path=f"/Visuals/lfoot_marker")
        ) 
        self.r_foot_marker = VisualizationMarkers(
            frame_marker_cfg.replace(prim_path=f"/Visuals/rfoot_marker")
        )
        self.goal_marker.visualize(self.targets_w, self.targets_quat_w)

    def print_robot_info(self):
        # Print body information
        print(f"[INFO]: Body Information...",
              f" Num links: {len(self._body_idx)}")
        for i in range(len(self.body_names)):
            print(
                f"Link {i}: "
                f"name={self.body_names[i]}, "
            )

        # Print joint information
        print(f"[INFO]: Joint Information...",
              f" Dofs: {len(self._joint_dof_idx)}")
        for i in range(len(self.joint_names)):
            print(
                f"Joint {i}: "
                f"name={self.joint_names[i]}, "
                f"lim=({self.joint_limits_lower[0, i].item():.3f}, "
                f"{self.joint_limits_upper[0, i].item():.3f})"
            )

    def _setup_scene(self):
        # robot
        self.robot = Articulation(self.cfg.robot)
        # ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_path=[self.cfg.terrain.prim_path])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.time_step += 1
        self.time_step %= self.period
        self.actions = actions.clone()
        self.actions = torch.clamp(self.actions, -1.0, 1.0)
        
    def _apply_action(self):
        action_scale = 1.0
        self.torque_cmd = action_scale * self.effort_limits[:, self._joint_dof_idx] * self.actions
        self.robot.set_joint_effort_target(self.torque_cmd, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        # Joint data
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        # Body data
        self.body_pos_w = self.robot.data.body_pos_w
        self.body_quat_w = self.robot.data.body_quat_w
        self.body_lin_vel_w = self.robot.data.body_lin_vel_w
        self.body_ang_vel_w = self.robot.data.body_ang_vel_w

        # COM
        self.com_w = self.robot.data.body_com_pos_w

        (
            self.joint_pos_scaled,
            self.yaw_error,
            self.roll,
            self.pitch,
            self.yaw,
            self.torso_heading_proj,
            self.torso_up_proj,
            self.torso_lin_vel_w,
            self.torso_ang_vel_w,
            self.up_body_up_proj,
            self.l_foot_side_proj,
            self.r_foot_side_proj,
            self.phase,
            self.l_foot_rel_pos_torso,
            self.r_foot_rel_pos_torso,
            self.l_arm_rel_pos_ub,
            self.r_arm_rel_pos_ub,
            self.com_a_w,
            self.com_a_rel,
            self.L

        ) = compute_intermediate_values(
            self.joint_pos,
            self.joint_limits_lower,
            self.joint_limits_upper,
            self.inv_start_rot,
            self.body_pos_w,
            self.body_quat_w,
            self.body_lin_vel_w,
            self.body_ang_vel_w,
            self.torso_idx,
            self.up_body_idx,
            self.l_foot_idx,
            self.r_foot_idx,
            self.l_arm_idx,
            self.r_arm_idx,
            self.mass,
            self.com_w,
            self.targets_w,
            self.basis_heading_vec,
            self.basis_up_vec,
            self.basis_side_vec,
            self.time_step,
            self.period,
        )
        
    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        phase_sin = torch.sin(self.phase).unsqueeze(-1)
        phase_cos = torch.cos(self.phase).unsqueeze(-1)
        obs = torch.cat(
            (
                self.joint_pos_scaled, #23
                self.joint_vel * self.joint_vel_scale, #23
                self.roll.unsqueeze(-1), #1
                self.pitch.unsqueeze(-1), #1
                self.yaw.unsqueeze(-1), #1
                self.torso_heading_proj.unsqueeze(-1), #1
                self.torso_up_proj.unsqueeze(-1), #1
                phase_sin, #1
                phase_cos, #1
                self.actions #23
            ),
            dim=-1
        )
        self.update_marker()
        return {"policy": obs}
    
    def update_marker(self):
        # Marker update
        self.base_marker.visualize(
            self.body_pos_w[:, self.torso_idx], 
            self.body_quat_w[:, self.torso_idx]
        )
        self.up_body_marker.visualize(
            self.body_pos_w[:, self.up_body_idx],
            self.body_quat_w[:, self.up_body_idx]
        )
        self.l_foot_marker.visualize(
            self.body_pos_w[:, self.l_foot_idx],
            self.body_quat_w[:, self.l_foot_idx]
        )
        self.r_foot_marker.visualize(
            self.body_pos_w[:, self.r_foot_idx],
            self.body_quat_w[:, self.r_foot_idx]
        )
        self.l_arm_marker.visualize(
            self.body_pos_w[:, self.l_arm_idx],
            self.body_quat_w[:, self.l_arm_idx]
        )
        self.r_arm_marker.visualize(
            self.body_pos_w[:, self.r_arm_idx],
            self.body_quat_w[:, self.r_arm_idx]
        )
    
    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
                self.joint_pos_scaled,
                self.joint_vel * self.joint_vel_scale,
                self.torso_lin_vel_w,
                self.torso_ang_vel_w,
                self.torso_heading_proj,
                self.torso_up_proj,
                self.up_body_up_proj,
                self.l_foot_side_proj,
                self.r_foot_side_proj,
                self.phase,
                self.foot_diff_amplitude,
                self.l_foot_rel_pos_torso,
                self.r_foot_rel_pos_torso,
                self.l_arm_rel_pos_ub,
                self.r_arm_rel_pos_ub,
                self.com_a_rel,
                self.L,
                self.actions,
                self.reset_terminated,
            )
        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.body_pos_w[:, self.torso_idx, 2] < self.termination_height
        return died, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.actions[env_ids] = 0

        to_target = self.targets_w[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0

        self.time_step[env_ids] = 0


@torch.jit.script
def compute_intermediate_values(
    joint_pos:torch.Tensor,
    joint_limits_lower:torch.Tensor,
    joint_limits_upper:torch.Tensor,
    inv_start_rot:torch.Tensor,
    body_pos_w:torch.Tensor,
    body_quat_w:torch.Tensor,
    body_lin_vel_w:torch.Tensor,
    body_ang_vel_w:torch.Tensor,
    torso_idx:int,
    up_body_idx:int,
    l_foot_idx:int,
    r_foot_idx:int,
    l_arm_idx:int,
    r_arm_idx:int,
    mass:torch.Tensor,
    com_w:torch.Tensor,
    targets:torch.Tensor,
    basis_heading_vec:torch.Tensor,
    basis_up_vec:torch.Tensor,
    basis_side_vec:torch.Tensor,
    time_step:torch.Tensor,
    period:float,
):
    ### JOINT POSITION SCALE ###
    joint_pos_scaled = 2.0 * (joint_pos - joint_limits_lower) / (joint_limits_upper - joint_limits_lower) - 1.0

    ### TORSO CALCULATION ###
    # The axes of torso frame in world frame
    # -y direction in torso frame -> x direction in world frame
    torso_pos_w = body_pos_w[:, torso_idx]
    torso_quat_w = body_quat_w[:, torso_idx]
    torso_lin_vel_w = body_lin_vel_w[:, torso_idx]
    torso_ang_vel_w = body_ang_vel_w[:, torso_idx]
    # torso direction  [world frame]
    torso_heading_vec_w = quat_rotate(torso_quat_w, basis_heading_vec) # rotate basis heading vector
    torso_up_vec_w = quat_rotate(torso_quat_w, basis_up_vec) # rotate basis up vector
    # target direction [world frame]
    # vector from current position
    to_target = targets - torso_pos_w
    to_target[:, 2] = 0.0
    to_target_dir = normalize(to_target)
    # heading alignment with target direciton
    torso_heading_proj = torch.sum(torso_heading_vec_w * to_target_dir, dim=-1)
    # up projection in world z-direction
    torso_up_proj = torso_up_vec_w[:, 2]
    # roll, pitch, yaw of torso
    roll = torch.atan2(torso_up_vec_w[:, 1], torso_up_vec_w[:, 2])
    pitch = torch.atan2(-torso_up_vec_w[:, 0], torso_up_vec_w[:, 2])
    yaw = torch.atan2(torso_heading_vec_w[:, 1], torso_heading_vec_w[:, 0])
    # heading angle(yaw)
    yaw_target = torch.atan2(to_target_dir[:, 1], to_target_dir[:, 0])
    yaw_error = yaw_target - yaw
    yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))

    ### UPPER BODY CALCULATION ###
    up_body_quat_w = body_quat_w[:, up_body_idx]
    # up vector projection
    up_body_up_vec_w = quat_rotate(up_body_quat_w, basis_up_vec)
    up_body_up_proj = up_body_up_vec_w[:, 2]

    ### FOOT CALCULATION ###
    # torso canonical frame
    torso_quat_can = quat_mul(inv_start_rot, torso_quat_w)
    #
    l_foot_pos_w = body_pos_w[:, l_foot_idx]
    l_foot_quat_w = body_quat_w[:, l_foot_idx]
    r_foot_pos_w = body_pos_w[:, r_foot_idx]
    r_foot_quat_w = body_quat_w[:, r_foot_idx]
    # foot local positions [world frame]
    l_foot_rel_pos_w = l_foot_pos_w - torso_pos_w
    l_foot_rel_pos_torso = quat_rotate_inverse(torso_quat_can, l_foot_rel_pos_w)
    r_foot_rel_pos_w = r_foot_pos_w - torso_pos_w
    r_foot_rel_pos_torso = quat_rotate_inverse(torso_quat_can, r_foot_rel_pos_w)
    # left foot direction [world frame]
    torso_side_vec_w = quat_rotate(torso_quat_w, basis_side_vec)
    l_foot_side_vec_w = quat_rotate(l_foot_quat_w, basis_side_vec)
    l_foot_side_proj = torch.sum(l_foot_side_vec_w * torso_side_vec_w, dim=-1)
    # right foot direction [world frame]
    r_foot_side_vec_w = quat_rotate(r_foot_quat_w, basis_side_vec)
    r_foot_side_proj = torch.sum(r_foot_side_vec_w * torso_side_vec_w, dim=-1)

    ### ARM CALCULATION ###
    # upper body canonical frame
    up_body_quat_can = quat_mul(inv_start_rot, up_body_quat_w)
    # position [world frame]
    l_arm_pos_w = body_pos_w[:, l_arm_idx]
    r_arm_pos_w = body_pos_w[:, r_arm_idx]
    # relatove position [world frame]
    l_arm_rel_pos_w = l_arm_pos_w - torso_pos_w
    r_arm_rel_pos_w = r_arm_pos_w - torso_pos_w
    # relative postiion [up body frame]
    l_arm_rel_pos_ub = quat_rotate_inverse(up_body_quat_can, l_arm_rel_pos_w)
    r_arm_rel_pos_ub = quat_rotate_inverse(up_body_quat_can, r_arm_rel_pos_w)

    ### GAIT PHASE ###
    phase = 2.0 * torch.pi * (time_step.float() / period)

    ### COM CONTROL ###
    mass = mass.unsqueeze(-1)
    weighted_pos = com_w * mass
    com_a_w = weighted_pos.sum(dim=1) / mass.sum(dim=1)
    com_a_rel = com_a_w - torso_pos_w

    ### ANGULAR MOMENTUM (Simple) ###
    r = body_pos_w - com_a_w.unsqueeze(1)
    p = mass * body_lin_vel_w
    L = torch.sum(torch.cross(r, p, dim=-1), dim=1)

    return (
        joint_pos_scaled,
        yaw_error,
        roll,
        pitch,
        yaw,
        torso_heading_proj,
        torso_up_proj,
        torso_lin_vel_w,
        torso_ang_vel_w,
        up_body_up_proj,
        l_foot_side_proj,
        r_foot_side_proj,
        phase,
        l_foot_rel_pos_torso,
        r_foot_rel_pos_torso,
        l_arm_rel_pos_ub,
        r_arm_rel_pos_ub,
        com_a_w,
        com_a_rel,
        L,
    )

@torch.jit.script
def compute_rewards(
    joint_pos_scaled:torch.Tensor,
    joint_vel_scaled:torch.Tensor,
    torso_lin_vel_w:torch.Tensor,
    torso_ang_vel_w:torch.Tensor,
    torso_heading_proj:torch.Tensor,
    torso_up_proj:torch.Tensor,
    up_body_up_proj:torch.Tensor,
    l_foot_side_proj:torch.Tensor,
    r_foot_side_proj:torch.Tensor,
    phase:torch.Tensor,
    foot_diff_amplitude:float,
    l_foot_rel_pos_torso:torch.Tensor,
    r_foot_rel_pos_torso:torch.Tensor,
    l_arm_rel_pos_ub:torch.Tensor,
    r_arm_rel_pos_ub:torch.Tensor,
    com_a_rel:torch.Tensor,
    L:torch.Tensor,
    actions:torch.Tensor,
    reset_terminated:torch.Tensor,
):
    # weights
    weight_dof_limit = 0.1
    weight_alive = 0.2
    weight_forward_vel = 0.8
    weight_lateral_vel = 0.5
    weight_T_heading = 0.8
    weight_T_up = 0.8
    weight_UB_up = 0.8
    weight_LF = 0.15
    weight_RF = 0.15
    weight_step_swing = 0.8
    weight_step_cross = 1.5
    weight_step_witdh = 1.0
    weight_arm_swing = 0.15
    weight_arm_cross = 0.3
    weight_arm_width = 0.3
    weight_com = 0.4
    weight_action = 0.03
    weight_energy = 0.02
    weight_L = 0.05
    weight_death = 2.0

    # [Penalty] Soft joint limits
    limit_margin = 0.98
    dof_limit_violation = torch.relu(torch.abs(joint_pos_scaled) - limit_margin)
    dof_at_limit_cost = torch.sum(dof_limit_violation, dim=-1)

    # [Reward] Alive
    alive_reward = torch.ones_like(dof_at_limit_cost)

    # [Reward] Velocity
    v_target = 1.0
    vel_err = torso_lin_vel_w[:, 0] - v_target
    forward_vel_reward = torch.exp(-30.0 * torch.square(vel_err))
    lateral_vel_penalty = torch.square(torso_lin_vel_w[:, 1])

    # [Reward] Torso heading direction (to target)
    one_reward = torch.ones_like(torso_heading_proj)
    T_heading_reward = torch.where(
        torso_heading_proj > 0.8, 
        one_reward, 
        torch.clamp(torso_heading_proj / 0.8, min=0.0)
    )

    # [Reward] Torso up direction (z-axis)
    T_up_reward = torch.clamp(torso_up_proj, min=0.0)

    # [Reward] Upper body up direction (z-axis)
    UB_up_reward = torch.clamp(up_body_up_proj, min=0.0)

    # [Reward] Left foot direction (saggital plane)
    LF_dir_reward = torch.clamp(l_foot_side_proj, min=0.0)

    # [Reward] Upper body direction (saggital plane)
    RF_dir_reward = torch.clamp(r_foot_side_proj, min=0.0)

    # [Reward] Footstep control
    feet_x_diff = l_foot_rel_pos_torso[:, 0] - r_foot_rel_pos_torso[:, 0]
    feet_x_desired = foot_diff_amplitude * torch.tanh(3.0 * torch.sin(phase))
    feet_dist_err = feet_x_diff - feet_x_desired
    step_reward = torch.exp(-50.0 * torch.square(feet_dist_err))

    # [Penalty] Foot cross
    cross_penalty = torch.relu(-l_foot_rel_pos_torso[:, 1]) + torch.relu(r_foot_rel_pos_torso[:, 1])

    # [Penalty] Foot width
    max_width = 0.2
    min_width = 0.05
    feet_y_dist = torch.abs(l_foot_rel_pos_torso[:, 1] - r_foot_rel_pos_torso[:, 1])
    width_penalty = torch.square(torch.relu(feet_y_dist - max_width)) \
                  + torch.square(torch.relu(min_width - feet_y_dist))

    # [Reward] Arm swing
    arm_x_diff = l_arm_rel_pos_ub[:, 0] - r_arm_rel_pos_ub[:, 0]
    arm_swing_reward = torch.clamp(
        -torch.sign(arm_x_diff) * torch.sign(feet_x_diff),
        min=0.0
    )

    # [Penalty] Arm cross
    arm_cross_penalty = torch.relu(-l_arm_rel_pos_ub[:, 1]) + torch.relu(r_arm_rel_pos_ub[:, 1])
    arm_center_penalty = torch.square(
        l_arm_rel_pos_ub[:, 0] + r_arm_rel_pos_ub[:, 0]
    )

    # [Penalty] Arm width
    min_width = 0.05
    arm_y_dist = torch.abs(l_arm_rel_pos_ub[:, 1] - r_arm_rel_pos_ub[:, 1])
    arm_width_penalty = torch.square(torch.relu(min_width - arm_y_dist))

    # [Reward] COM
    com_x_target = 0.02
    com_x_diff = com_a_rel[:, 0] - com_x_target
    com_forward_reward = torch.exp(-50.0 * torch.square(com_x_diff))

    # [Penalty] Angular momentum
    yaw_momentum_penalty = torch.square(torch.relu(torch.abs(L[:, 2]) - 0.5))

    # [Penalty] Action
    actions_cost = torch.mean(actions**2, dim=-1)

    # [Penalty] Energy consumption
    energy_cost = torch.mean(torch.abs(actions * joint_vel_scaled), dim=-1)

    total_reward = (
        - weight_dof_limit * dof_at_limit_cost
        + weight_alive * alive_reward
        + weight_forward_vel * forward_vel_reward
        - weight_lateral_vel * lateral_vel_penalty
        + weight_T_heading * T_heading_reward
        + weight_T_up * T_up_reward
        + weight_UB_up * UB_up_reward
        + weight_LF * LF_dir_reward
        + weight_RF * RF_dir_reward
        + weight_step_swing * step_reward
        - weight_step_cross * cross_penalty
        - weight_step_witdh * width_penalty
        + weight_arm_swing * arm_swing_reward
        - weight_arm_cross * (arm_cross_penalty + arm_center_penalty)
        - weight_arm_width * arm_width_penalty
        + weight_com * com_forward_reward
        - weight_L * yaw_momentum_penalty
        - weight_action * actions_cost
        - weight_energy * energy_cost
    )

    # [Penalty] Death penalty for fallen agents
    death_cost = -weight_death
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)

    ### LOG ###
    # print("forward", forward_vel_reward.mean().item())
    # print("step", step_reward.mean().item())
    # print("arm_swing", arm_swing_reward.mean().item())
    # print("cross", cross_penalty.mean().item())
    # print("arm_cross", arm_cross_penalty.mean().item())
    # print("action", actions_cost.mean().item())
    # print("energy", energy_cost.mean().item())
    # print("yaw_L", yaw_momentum_penalty.mean().item())
    
    return total_reward
    