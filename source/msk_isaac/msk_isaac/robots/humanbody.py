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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
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
    episode_length_s = 10.0
    action_space = 23
    observation_space = 10 + 3 * action_space
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
        num_envs=500, env_spacing=10.0, replicate_physics=False, clone_in_fabric=False
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
        self.torso_pos_w = self.robot.data.root_pos_w
        self.torso_quat_w = self.robot.data.root_quat_w
        self.inv_start_rot = quat_conjugate(self.torso_quat_w)

        # Fixed Frame (world) : pos = (x y z), quat = (w x y z)
        self.origin_pos_w = torch.tensor([0,0,0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.origin_quat_w = torch.tensor([1,0,0,0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))

        # Joint properties
        self._joint_dof_idx, self.joint_names = self.robot.find_joints(".*")
        self.joint_limits = self.robot.data.joint_pos_limits
        self.joint_limits_lower = self.joint_limits[:, :, 0]
        self.joint_limits_upper = self.joint_limits[:, :, 1]
        self.effort_limits = self.robot.data.joint_effort_limits
        self.motor_effort_ratio = torch.ones_like(self.effort_limits, device=self.sim.device)

        # EE Body indexes for easy control
        up_body_ids, _ = self.robot.find_bodies("upperbody")
        self.up_body_idx = up_body_ids[0]
        l_arm_ids, _ = self.robot.find_bodies("left_lowerarm")
        self.l_arm_idx = l_arm_ids[0]
        r_arm_ids, _ = self.robot.find_bodies("right_lowerarm")
        self.r_arm_idx = r_arm_ids[0]
        l_leg_ids, _ = self.robot.find_bodies("left_tib")
        self.l_leg_idx = l_leg_ids[0]
        r_leg_ids, _ = self.robot.find_bodies("right_tib")
        self.r_leg_idx = r_leg_ids[0]
        l_foot_ids, _ = self.robot.find_bodies("left_foot")
        self.l_foot_idx = l_foot_ids[0]
        r_foot_ids, _ = self.robot.find_bodies("right_foot")
        self.r_foot_idx = r_foot_ids[0]
        
        # Target
        # potential : negative 2-norm of base-to-target vector
        self.targets = torch.tensor([100, 0, 0],dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.targets += self.scene.env_origins
        self.targets_quat = self.origin_quat_w.clone()
        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        
        # Heading and up vectors
        self.heading_vec = torch.tensor([0, -1, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))

        # References for heading and up vectors
        self.basis_heading_vec = self.heading_vec.clone()
        self.basis_up_vec = self.up_vec.clone()
        self.basis_side_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))

        # Time step control
        self.time_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim.device)

        ### Gait control ###
        target_vel = 1.0 # m/s
        rl_dt = self.cfg.decimation*self.cfg.sim.dt # rl step duration
        # step frequency [steps/s]
        f_min = 1.4
        f_max = 2.0
        v_ref = 1.0
        speed_ratio = max(min(target_vel / v_ref, 1.0), 0.0)
        f_steps = f_min + (f_max -f_min) * speed_ratio
        # step length
        step_length = target_vel / f_steps
        self.foot_diff_amplitude = 0.5 * step_length
        # gait period
        self.period = max(10, int(round(1.0 / (f_steps * rl_dt))))
        
        # COM calculation
        self.mass = self.robot.root_physx_view.get_masses().to(self.sim.device)

        # Weights and Scales
        self.termination_height: float = 0.7
        self.joint_vel_scale: float = 0.1

        self.alive_reward_scale: float = 2.0
        self.death_cost: float = -1.0
        self.heading_weight: float = 0.5
        self.up_weight: float = 0.5
        self.energy_cost_scale: float = 0.05
        self.actions_cost_scale: float = 0.01
        
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
        self.goal_marker.visualize(self.targets, self.targets_quat)

        # Print joint information
        print(f"[INFO]: Joint Information...",
              f" Dofs: {len(self._joint_dof_idx)}")
        for i in range(len(self.joint_names)):
            print(
                f"Joint {i}: "
                f"idx={self._joint_dof_idx[i]}, "
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
        self.actions = actions.clone()
        self.actions = torch.clamp(self.actions, -1.0, 1.0)
        
    def _apply_action(self):
        action_scale = 1.0
        forces = action_scale * self.effort_limits[:, self._joint_dof_idx] * self.actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        # Torso (articulation root) data
        self.torso_pos_w = self.robot.data.root_pos_w
        self.torso_quat_w = self.robot.data.root_quat_w
        self.torso_lin_vel_w = self.robot.data.root_lin_vel_w
        self.torso_ang_vel_w = self.robot.data.root_ang_vel_w
        
        # Body data
        # local: (x, y, z) -> world: (y, -x, z)
        self.up_body_pos_w = self.robot.data.body_pos_w[:, self.up_body_idx]
        self.up_body_quat_w = self.robot.data.body_quat_w[:, self.up_body_idx]
        self.l_arm_pos_w = self.robot.data.body_pos_w[:, self.l_arm_idx]
        self.l_arm_quat_w = self.robot.data.body_quat_w[:, self.l_arm_idx]
        self.r_arm_pos_w = self.robot.data.body_pos_w[:, self.r_arm_idx]
        self.r_arm_quat_w = self.robot.data.body_quat_w[:, self.r_arm_idx]
        self.l_leg_pos_w = self.robot.data.body_pos_w[:, self.l_leg_idx]
        self.l_leg_quat_w = self.robot.data.body_quat_w[:, self.l_leg_idx]
        self.r_leg_pos_w = self.robot.data.body_pos_w[:, self.r_leg_idx]
        self.r_leg_quat_w = self.robot.data.body_quat_w[:, self.r_leg_idx]
        self.l_foot_pos_w = self.robot.data.body_pos_w[:, self.l_foot_idx]
        self.l_foot_quat_w = self.robot.data.body_quat_w[:, self.l_foot_idx]
        self.r_foot_pos_w = self.robot.data.body_pos_w[:, self.r_foot_idx]
        self.r_foot_quat_w = self.robot.data.body_quat_w[:, self.r_foot_idx]

        # Joint data
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Marker update
        self.base_marker.visualize(self.torso_pos_w, self.torso_quat_w)
        self.up_body_marker.visualize(self.up_body_pos_w, self.up_body_quat_w)
        self.l_arm_marker.visualize(self.l_arm_pos_w, self.l_arm_quat_w)
        self.r_arm_marker.visualize(self.r_arm_pos_w, self.r_arm_quat_w)
        self.l_foot_marker.visualize(self.l_foot_pos_w, self.l_foot_quat_w)
        self.r_foot_marker.visualize(self.r_foot_pos_w, self.r_foot_quat_w)

        # COM
        self.com_w = self.robot.data.body_com_pos_w

        (
            self.joint_pos_scaled,
            self.potentials,
            self.prev_potentials,
            self.angle_to_target,
            self.roll, # world x-axis
            self.pitch, # world y-axis
            self.yaw, # world z-axis
            self.torso_heading_proj, # target vector proj
            self.torso_up_proj, # local z-axis proj
            self.lin_vel_loc, # local linear vel
            self.ang_vel_loc, # local angular vel
            self.up_body_up_proj,
            self.l_foot_side_proj, # local x-axis proj
            self.r_foot_side_proj,
            self.phase, # gait phase
            self.l_foot_pos_loc,
            self.r_foot_pos_loc,
            self.com_whole, # com
            self.com_loc,

        ) = compute_intermediate_values(
            self.joint_pos,
            self.joint_limits_lower,
            self.joint_limits_upper,
            self.targets,
            self.potentials,
            self.prev_potentials,
            self.basis_heading_vec,
            self.basis_up_vec,
            self.basis_side_vec,
            self.inv_start_rot,
            self.torso_pos_w,
            self.torso_quat_w,
            self.torso_lin_vel_w,
            self.torso_ang_vel_w,
            self.up_body_quat_w,
            self.l_foot_pos_w,
            self.l_foot_quat_w,
            self.r_foot_pos_w,
            self.r_foot_quat_w,
            self.mass,
            self.com_w,
            self.time_step,
            self.period,
            self.cfg.sim.dt,
        )

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        # print(self.joint_pos_scaled.shape)
        # print(self.joint_vel.shape)
        # print(self.potentials.shape)
        # print(self.roll.shape)
        # print(self.pitch.shape)
        # print(self.yaw.shape)
        # print(self.torso_heading_proj.shape)
        # print(self.torso_up_proj.shape)
        # print(self.up_body_up_proj.shape)
        # print(self.l_foot_side_proj.shape)
        # print(self.r_foot_side_proj.shape)
        # print(self.phase.shape)
        # print(self.com_whole.shape)
        # print(self.actions.shape)
        # exit()
        obs = torch.cat(
            (
                self.joint_pos_scaled,
                self.joint_vel * self.joint_vel_scale,
                self.roll.unsqueeze(-1),
                self.pitch.unsqueeze(-1),
                self.yaw.unsqueeze(-1),
                self.torso_heading_proj.unsqueeze(-1),
                self.torso_up_proj.unsqueeze(-1),
                self.up_body_up_proj.unsqueeze(-1),
                self.phase.unsqueeze(-1),
                self.com_whole,
                self.actions
            ),
            dim=-1
        )
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
                self.joint_pos_scaled,
                self.joint_vel * self.joint_vel_scale,
                self.potentials,
                self.prev_potentials,
                self.torso_lin_vel_w,
                self.torso_ang_vel_w,
                self.torso_heading_proj,
                self.torso_up_proj,
                self.up_body_up_proj,
                self.l_foot_side_proj,
                self.r_foot_side_proj,
                self.phase,
                self.foot_diff_amplitude,
                self.l_foot_pos_loc,
                self.r_foot_pos_loc,
                self.com_loc,
                self.actions,
                self.reset_terminated,
            )
        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
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

        self.time_step[env_ids] = 0


@torch.jit.script
def compute_intermediate_values(
    joint_pos:torch.Tensor,
    joint_limits_lower:torch.Tensor,
    joint_limits_upper:torch.Tensor,
    targets:torch.Tensor,
    potentials:torch.Tensor,
    prev_potentials:torch.Tensor,
    basis_heading_vec:torch.Tensor,
    basis_up_vec:torch.Tensor,
    basis_side_vec:torch.Tensor,
    inv_start_rot:torch.Tensor,
    torso_pos_w:torch.Tensor,
    torso_quat_w:torch.Tensor,
    torso_lin_vel_w:torch.Tensor,
    torso_ang_vel_w:torch.Tensor,
    up_body_quat_w:torch.Tensor,
    l_foot_pos_w:torch.Tensor,
    l_foot_quat_w:torch.Tensor,
    r_foot_pos_w:torch.Tensor,
    r_foot_quat_w:torch.Tensor,
    mass:torch.Tensor,
    com_w:torch.Tensor,
    time_step:torch.Tensor,
    period:float,
    dt: float,
):
    
    ### JOINT POSITION SCALE ###
    joint_pos_scaled = 2.0 * (joint_pos - joint_limits_lower) / (joint_limits_upper - joint_limits_lower) - 1.0

    ### TORSO CALCULATION ###
    # The axes of torso frame in world frame
    # -y direction in torso frame -> x direction in world frame
    torso_quat_local = quat_mul(inv_start_rot, torso_quat_w)
    torso_heading_vec = quat_rotate(torso_quat_w, basis_heading_vec)
    torso_up_vec = quat_rotate(torso_quat_w, basis_up_vec)
    
    # target vector
    to_target = targets - torso_pos_w
    to_target[:, 2] = 0.0

    # target direction
    to_target_dir = normalize(to_target)

    # heading alignment
    torso_heading_proj = torch.sum(torso_heading_vec * to_target_dir, dim=-1)

    # up alignment
    torso_up_proj = torso_up_vec[:, 2]

    # potential field
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    # roll, pitch, yaw of torso
    # local: (x,y,z) -> world: (y,-x,z)
    roll, pitch, yaw = quat_to_euler(torso_quat_local)
    temp = roll.clone()
    roll = pitch.clone()
    pitch = -temp.clone()

    # heading angle(yaw)
    heading = torch.atan2(to_target_dir[:, 1], to_target_dir[:, 0])
    angle_to_target = heading - yaw
    angle_to_target = torch.atan2(torch.sin(angle_to_target), torch.cos(angle_to_target))

    # local velocities
    lin_vel_loc = quat_rotate_inverse(torso_quat_local, torso_lin_vel_w)
    ang_vel_loc = quat_rotate_inverse(torso_quat_local, torso_ang_vel_w)

    ### UPPER BODY CALCULATION ###
    up_body_up_vec = quat_rotate(up_body_quat_w, basis_up_vec)
    up_body_up_proj = up_body_up_vec[:, 2]

    ### FOOT CALCULATION ###
    # left foot direction
    l_foot_side_vec = quat_rotate(l_foot_quat_w, basis_side_vec)
    l_foot_side_proj = l_foot_side_vec[:, 1]
    # right foot direction
    r_foot_side_vec = quat_rotate(r_foot_quat_w, basis_side_vec)
    r_foot_side_proj = r_foot_side_vec[:, 1]

    ### GAIT CONTROL ###
    phase = 2.0 * torch.pi * (time_step.float() / period)
    l_foot_pos_loc = quat_rotate_inverse(torso_quat_local, l_foot_pos_w - torso_pos_w)
    r_foot_pos_loc = quat_rotate_inverse(torso_quat_local, r_foot_pos_w - torso_pos_w)

    ### COM CONTROL ###
    mass = mass.unsqueeze(-1)
    weighted_pos = com_w * mass
    com_whole = weighted_pos.sum(dim=1) / mass.sum()
    com_loc = quat_rotate_inverse(torso_quat_local, com_whole - torso_pos_w)

    return (
        joint_pos_scaled,
        potentials,
        prev_potentials,
        angle_to_target,
        roll,
        pitch,
        yaw,
        torso_heading_proj,
        torso_up_proj,
        lin_vel_loc,
        ang_vel_loc,
        up_body_up_proj,
        l_foot_side_proj,
        r_foot_side_proj,
        phase,
        l_foot_pos_loc,
        r_foot_pos_loc,
        com_whole,
        com_loc,
    )

@torch.jit.script
def compute_rewards(
    joint_pos_scaled: torch.Tensor,
    joint_vel_scaled: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    lin_vel_w: torch.Tensor,
    ang_vel_w: torch.Tensor,
    torso_heading_proj: torch.Tensor,
    torso_up_proj: torch.Tensor,
    up_body_up_proj: torch.Tensor,
    l_foot_side_proj: torch.Tensor,
    r_foot_side_proj: torch.Tensor,
    phase: torch.Tensor,
    foot_diff_amplitude:float,
    l_foot_pos_loc: torch.Tensor,
    r_foot_pos_loc: torch.Tensor,
    com_loc: torch.Tensor,
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    # weights
    weight_dof_limit = 0.2
    weight_progress = 0.0
    weight_alive = 0.5
    weight_forward_vel = 1.0
    weight_lateral_vel = 0.1
    weight_T_heading = 0.5
    weight_T_up = 0.3
    weight_UB_up = 0.3
    weight_LF = 0.1
    weight_RF = 0.1
    weight_step = 1.0
    weight_cross = 0.3
    weight_step_witdh = 0.3
    weight_com = 0.1
    weight_action = 0.05
    weight_energy = 0.02
    weight_death = 2.0

    # [Penalty] Soft joint limits
    limit_margin = 0.98
    dof_limit_violation = torch.relu(torch.abs(joint_pos_scaled) - limit_margin)
    dof_at_limit_cost = torch.sum(dof_limit_violation, dim=-1)

    # [Reward] Progess
    progress_reward = potentials - prev_potentials

    # [Reward] Alive
    alive_reward = torch.ones_like(potentials)

    # [Reward] Velocity
    v_target = 1.0
    vel_err = lin_vel_w[:, 0] - v_target
    forward_vel_reward = torch.exp(-4.0 * torch.square(vel_err))
    lateral_vel_penalty = torch.square(lin_vel_w[:, 1])

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
    feet_x_diff = l_foot_pos_loc[:, 0] - r_foot_pos_loc[:, 0]
    feet_x_desired = foot_diff_amplitude * torch.tanh(3.0 * torch.sin(phase))
    feet_dist_err = feet_x_diff - feet_x_desired
    step_reward = torch.exp(-50.0 * torch.square(feet_dist_err))

    # [Penalty] Foot cross
    cross_penalty = torch.relu(-l_foot_pos_loc[:, 1]) + torch.relu(r_foot_pos_loc[:, 1])

    # [Reward] Foot width
    target_width = 0.12
    foot_y_dist = torch.abs(l_foot_pos_loc[:, 1] - r_foot_pos_loc[:, 1])
    width_reward = torch.exp(-50.0 * torch.square(foot_y_dist - target_width))

    # [Reward] COM
    com_x_target = 0.02
    com_x_diff = com_loc[:, 0] - com_x_target
    com_forward_reward = torch.exp(-50.0 * torch.square(com_x_diff))

    # [Penalty] Energy consumption
    actions_cost = torch.sum(actions**2, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions *joint_vel_scaled), dim=-1)

    total_reward = (
        - weight_dof_limit * dof_at_limit_cost
        + weight_progress * progress_reward
        + weight_alive * alive_reward
        + weight_forward_vel * forward_vel_reward
        - weight_lateral_vel * lateral_vel_penalty
        + weight_T_heading * T_heading_reward
        + weight_T_up * T_up_reward
        + weight_UB_up * UB_up_reward
        + weight_LF * LF_dir_reward
        + weight_RF * RF_dir_reward
        + weight_step * step_reward
        - weight_cross * cross_penalty
        + weight_step_witdh * width_reward
        # + weight_com * com_forward_reward
        - weight_action * actions_cost
        - weight_energy * electricity_cost
    )

    # [Penalty] Death penalty for fallen agents
    death_cost = -weight_death
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
    return total_reward
    