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

# Get asset path
import os, math
ASSET_PATH = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'assets')

# Data class
from dataclasses import dataclass

# Custom packages
from custom_math.quaternion import quat_conjugate, quat_mul, quat_rotate, quat_rotate_inverse, quat_to_euler
from custom_math.math import normalize, normalize_angle

#######################
# Robot Configuration #
#######################
HUMANBODY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/mskdyros/msk_ws/msk_isaac/assets/isaac_humanbody_description/urdf/isaac_humanbody_float_base/isaac_humanbody.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        rot=(math.sin(math.pi/4), 0.0, 0.0, math.cos(math.pi/4)),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "waist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["waist_pitch"],
            effort_limit_sim=250.0,
            velocity_limit_sim=10.0,
            stiffness=1500.0,
            damping=100.0,
        ),
        "waist_roll": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=1500.0,
            damping=100.0,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw"],
            effort_limit_sim=60.0,
            velocity_limit_sim=10.0,
            stiffness=1500.0,
            damping=100.0,
        ),
        "shoulder_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch"],
            effort_limit_sim=40.0,
            velocity_limit_sim=10.0,
            stiffness=1500.0,
            damping=100.0,
        ),
        "shoulder_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_roll"],
            effort_limit_sim=35.0,
            velocity_limit_sim=10.0,
            stiffness=1500.0,
            damping=100.0,
        ),
        "shoulder_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_yaw"],
            effort_limit_sim=30.0,
            velocity_limit_sim=10.0,
            stiffness=1500.0,
            damping=100.0,
        ),
        "elbow_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_elbow_pitch"],
            effort_limit_sim=45.0,
            velocity_limit_sim=10.0,
            stiffness=1500.0,
            damping=100.0,
        ),
        "thigh_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_pitch"],
            effort_limit_sim=150.0,
            velocity_limit_sim=10.0,
            stiffness=1500.0,
            damping=100.0,
        ),
        "thigh_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_roll"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=1500.0,
            damping=100.0,
        ),
        "thigh_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_yaw"],
            effort_limit_sim=70.0,
            velocity_limit_sim=10.0,
            stiffness=1500.0,
            damping=100.0,
        ),
        "knee_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_pitch"],
            effort_limit_sim=150.0,
            velocity_limit_sim=10.0,
            stiffness=1500.0,
            damping=100.0,
        ),
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=1500.0,
            damping=100.0,
        ),
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll"],
            effort_limit_sim=60.0,
            velocity_limit_sim=10.0,
            stiffness=1500.0,
            damping=100.0,
        ),
    },
)

# Robot actuator parameters setup for torque control
for _, actuator_cfg in HUMANBODY_CFG.actuators.items():
    actuator_cfg.stiffness = 0.0
    actuator_cfg.damping = 0.0


##################################
# Environment Configuration (RL) #
##################################
@configclass
class HumanbodyEnvCfg(DirectRLEnvCfg):
    # required
    decimation = 2
    episode_length_s = 15.0
    action_space = 23
    observation_space = 12 + 3 * action_space
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
        num_envs=500, env_spacing=4.0, replicate_physics=False, clone_in_fabric=False
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
        # init once (not updated)
        self._joint_dof_idx, self.joint_names = self.robot.find_joints(".*")
        self.root_quat_w = self.robot.data.root_quat_w
        self.inv_start_rot = quat_conjugate(self.root_quat_w)
        self.joint_limits = self.robot.data.joint_pos_limits
        self.joint_limits_lower = self.joint_limits[:, :, 0]
        self.joint_limits_upper = self.joint_limits[:, :, 1]
        self.effort_limits = self.robot.data.joint_effort_limits
        self.motor_effort_ratio = torch.ones_like(self.effort_limits, device=self.sim.device)

        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        self.targets = torch.tensor([3, 0, 0],dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.ang_vel_scale = 0.25
        self.joint_vel_scale = 0.1
        self.heading_weight = 0.5
        self.up_weight = 0.1
        self.actions_cost_scale = 0.01
        self.alive_reward_scale = 2.0
        self.energy_cost_scale = 0.05
        self.death_cost = -1
        self.termination_height = 0.7

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
        self.actions = actions.clone()
        self.actions = torch.clamp(self.actions, -1, 1)

    def _apply_action(self):
        action_scale = 1.0
        forces = action_scale * self.effort_limits[:, self._joint_dof_idx] * self.actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        # torso (articulation root) data
        self.torso_pos_w = self.robot.data.root_pos_w
        self.torso_quat_w = self.robot.data.root_quat_w
        self.torso_lin_vel_w = self.robot.data.root_lin_vel_w
        self.torso_ang_vel_w = self.robot.data.root_ang_vel_w

        # joint data
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.lin_vel_loc,
            self.ang_vel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.joint_pos_scaled,
            self.prev_potentials,
            self.potentials
        ) = compute_intermediate_values(
            self.targets,
            self.torso_pos_w,
            self.torso_quat_w,
            self.torso_lin_vel_w,
            self.torso_ang_vel_w,
            self.joint_pos,
            self.joint_limits_lower,
            self.joint_limits_upper,
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt
        )

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.torso_pos_w[:, 2].view(-1, 1), # (N,1)
                self.lin_vel_loc, # (N, 3)
                self.ang_vel_loc * self.ang_vel_scale, # (N, 3)
                normalize_angle(self.yaw).unsqueeze(-1), # (N,1)
                normalize_angle(self.roll).unsqueeze(-1), # (N,1)
                normalize_angle(self.angle_to_target).unsqueeze(-1), # (N,1)
                self.up_proj.unsqueeze(-1), # (N,1)
                self.heading_proj.unsqueeze(-1), # (N,1)
                self.joint_pos_scaled, # (N,23)
                self.joint_vel * self.joint_vel_scale, # (N,23)
                self.actions # (N,23)
            ),
            dim=-1
        )
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
                self.actions,
                self.reset_terminated,
                self.up_weight,
                self.heading_weight,
                self.heading_proj,
                self.up_proj,
                self.joint_vel,
                self.joint_pos_scaled,
                self.potentials,
                self.prev_potentials,
                self.actions_cost_scale,
                self.energy_cost_scale,
                self.joint_vel_scale,
                self.death_cost,
                self.alive_reward_scale,
                self.motor_effort_ratio
            )
        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.torso_pos_w[:, 2] < self.termination_height
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

        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt

        self._compute_intermediate_values()


@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_pos_w: torch.Tensor,
    torso_quat_w: torch.Tensor,
    torso_lin_vel_w: torch.Tensor,
    torso_ang_vel_w: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_limits_lower: torch.Tensor,
    joint_limits_upper: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    # normalize rotation based on init rotation
    torso_quat_i = quat_mul(inv_start_rot, torso_quat_w)

    # The axes of torso frame
    # local -> world
    heading_vec = quat_rotate(torso_quat_i, basis_vec0)
    up_vec = quat_rotate(torso_quat_i, basis_vec1)

    # target vector
    to_target = targets - torso_pos_w
    to_target[:, 2] = 0.0

    # target direction
    to_target_dir = normalize(to_target)

    # heading alignment
    heading_proj = torch.sum(heading_vec * to_target_dir, dim=-1)

    # up alignment
    up_proj = up_vec[:, 2]

    # local velocities
    # world -> local
    vel_loc = quat_rotate_inverse(torso_quat_i, torso_lin_vel_w)
    ang_vel_loc = quat_rotate_inverse(torso_quat_i, torso_ang_vel_w)

    # roll, pitch, yaw of torso
    roll, pitch, yaw = quat_to_euler(torso_quat_i)

    # heading angle(yaw)
    heading = torch.atan2(to_target_dir[:, 1], to_target_dir[:, 0])
    angle_to_target = heading - yaw
    angle_to_target = torch.atan2(torch.sin(angle_to_target), torch.cos(angle_to_target))

    # joint pos scaled
    joint_pos_scaled = 2.0 * (joint_pos - joint_limits_lower) / (joint_limits_upper - joint_limits_lower) - 1.0
    
    # potential field
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        ang_vel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        joint_pos_scaled,
        prev_potentials,
        potentials,
    )

@torch.jit.script
def compute_rewards(
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    actions_cost_scale: float,
    energy_cost_scale: float,
    dof_vel_scale: float,
    death_cost: float,
    alive_reward_scale: float,
    motor_effort_ratio: torch.Tensor,
):
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions**2, dim=-1)
    electricity_cost = torch.sum(
        torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
        dim=-1,
    )

    # dof at limit cost
    dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost
    )
    # adjust reward for fallen agents
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
    return total_reward
    