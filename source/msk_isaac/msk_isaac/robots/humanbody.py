"""
Docstring for robots.humanbody
"""
# Import isaaclab packages
import torch, os
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
            damping=0.0,
        ),
        "waist_roll": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw"],
            effort_limit_sim=60.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "shoulder_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch"],
            effort_limit_sim=40.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "shoulder_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_roll"],
            effort_limit_sim=35.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "shoulder_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_yaw"],
            effort_limit_sim=30.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "elbow_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_elbow_pitch"],
            effort_limit_sim=45.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "thigh_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_pitch"],
            effort_limit_sim=150.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "thigh_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_roll"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "thigh_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_thigh_yaw"],
            effort_limit_sim=70.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "knee_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_pitch"],
            effort_limit_sim=150.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll"],
            effort_limit_sim=60.0,
            velocity_limit_sim=10.0,
            stiffness=0.0,
            damping=0.0,
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
    action_space = 15
    num_dofs = 23
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
            dynamic_friction=0.8,
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
        # simulation time
        self.sim_time_step = 0
        self.rl_dt = self.cfg.decimation*self.cfg.sim.dt # rl step duration
        self.env_time_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim.device)

        ### Robot Params ###
        # Torso (base) Frame
        self.torso_pos_w = self.robot.data.root_pos_w
        self.torso_quat_w = self.robot.data.root_quat_w
        self.init_torso_pos_w = self.torso_pos_w.clone()
        self.init_torso_quat_w = self.torso_quat_w.clone()
        self.inv_start_rot = quat_conjugate(self.init_torso_quat_w)

        # Joint properties
        self._joint_dof_ids, self.joint_names = self.robot.find_joints(".*")
        self.joint_limits = self.robot.data.joint_pos_limits
        self.joint_limits_lower = self.joint_limits[:, :, 0]
        self.joint_limits_upper = self.joint_limits[:, :, 1]
        self.effort_limits = self.robot.data.joint_effort_limits
        self.motor_effort_ratio = torch.ones_like(self.effort_limits, device=self.sim.device)
        self.UB_joint_ids, _ = self.robot.find_joints([".*shoulder.*", ".*elbow.*"])
        self.LB_joint_dof_ids, _ = self.robot.find_joints([".*waist.*", ".*thigh.*", ".*knee.*", ".*ankle.*"])

        # Body
        self._body_ids, self.body_names = self.robot.find_bodies(".*") 

        # Body indexes
        name_to_idx = {name: i for i, name in enumerate(self.robot.body_names)}
        self.pelvis_idx = name_to_idx["base_link"]
        self.torso_idx = name_to_idx["upperbody"]
        self.l_foot_idx = name_to_idx["left_foot"]
        self.r_foot_idx = name_to_idx["right_foot"]
        self.l_arm_idx = name_to_idx["left_lowerarm"]
        self.r_arm_idx = name_to_idx["right_lowerarm"]

        self.UB_link_ids, _ = self.robot.find_bodies(["upperbody", ".*upperarm", ".*lowerarm"])
        self.UB_link_ids_tensor = torch.tensor(self.UB_link_ids, dtype=torch.long, device=self.sim.device)
        
        ### Targets ###
        # Target vectors
        self.targets_w = torch.tensor([0, 0, 0],dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.targets_w += self.scene.env_origins
        self.targets_quat_w = torch.tensor([1,0,0,0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.init_targets_w = self.targets_w.clone()

        # Direction vectors in torso frame
        self.basis_heading_vec = torch.tensor([0, -1, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.basis_up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.basis_side_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        
        ### Gait control ###
        # z offset
        torso_pos_w = self.robot.data.body_pos_w[:, self.torso_idx]
        l_foot_pos_w = self.robot.data.body_pos_w[:, self.l_foot_idx]
        r_foot_pos_w = self.robot.data.body_pos_w[:, self.r_foot_idx]
        self.foot_z_offset = torso_pos_w[:, 2] - 0.5 * (l_foot_pos_w[:, 2] + r_foot_pos_w[:, 2])

        # time step for gait phase
        self.gait_time_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.sim.device)
        
        # target forward velocity
        self.target_vel = 1.0 # m/s
        
        # speed ratio
        v_ref = 1.0
        speed_ratio = max(min(self.target_vel / v_ref, 1.0), 0.0)

        # stride frequency [Hz] : one full cycle for one step
        f_stride_min = 0.8
        f_stride_max = 1.2
        self.f_stride = f_stride_min + (f_stride_max - f_stride_min) * speed_ratio

        # step length [m] : one foot step distance
        # step_length_min = 0.12
        # step_length_max = 0.28
        # self.step_length = step_length_min + (step_length_max - step_length_min) * speed_ratio
        self.step_length = v_ref / self.f_stride
        self.step_length /= 2.0

        # gait period [time steps / stride]
        self.period = max(10, int(round(1.0 / (self.f_stride * self.rl_dt))))

        # Potentials
        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        
        ### Dynamics ###
        # Link mass
        self.mass = self.robot.root_physx_view.get_masses().to(self.sim.device)

        ### Others ###
        self.termination_height: float = 0.8
        self.joint_vel_scale: float = 0.1

        ### Logs ###
        self.log_path = "/tmp/reward_log.csv"

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write(
                    "step,"
                    "p_joint_limits,p_actions,p_energy,"
                    "r_pelv_vel_x,r_yaw,"
                    "r_torso_head,r_torso_up,"
                    "r_pelv_head,r_pelv_up,"
                    "r_lfoot_side,r_rfoot_side,"
                    "r_lfoot_x,r_lfoot_y,r_lfoot_z,"
                    "r_rfoot_x,r_rfoot_y,r_rfoot_z,"
                    "r_alive,r_progress,"
                    "r_com_x,"
                    "p_L,"
                    "total_reward\n"
                )
                        
        ### Initialization ###
        self.print_robot_info()
        self.setup_visual_markers()
        self._compute_intermediate_values()


    def setup_visual_markers(self):
        # Markers for visualization
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
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
        
        # Optional markers 
        self.markers = []
        frame_marker_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        for i in range(1):
            new_marker = VisualizationMarkers(
                frame_marker_cfg.replace(prim_path=f"/Visuals/extra_marker_{i}")
            )
            self.markers.append(new_marker)


    def print_robot_info(self):
        # Print body information
        print(f"[INFO]: Body Information...",
              f" Num links: {len(self._body_ids)}")
        for i in range(len(self.body_names)):
            print(
                f"Link {i}: "
                f"name={self.body_names[i]}, "
            )

        # Print joint information
        print(f"[INFO]: Joint Information...",
              f" Dofs: {len(self._joint_dof_ids)}")
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
        # Action clamping
        self.actions = actions.clone()
        self.actions = torch.clamp(self.actions, -1.0, 1.0)
    

    def update_target(self):
        # Update target point [world frame]
        self.targets_w[:, 0] += self.target_vel * self.rl_dt
        # Relative position to torso position
        self.targets_rel_w = self.targets_w - self.body_pos_w[:, self.torso_idx]
        self.targets_rel_w[:, 2] = 0 # ignore height


    def update_marker(self):
        self.base_marker.visualize(
            self.body_pos_w[:, self.pelvis_idx], 
            self.body_quat_w[:, self.pelvis_idx]
        )
        self.up_body_marker.visualize(
            self.body_pos_w[:, self.torso_idx],
            self.body_quat_w[:, self.torso_idx]
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
        self.goal_marker.visualize(
            self.targets_w,
            self.targets_quat_w
        )
        self.markers[0].visualize(
            self.com_w,
            self.targets_quat_w
        )


    def _apply_action(self):
        # Action control
        action_scale = 1.0
        # Command torques calculation
        # Only consider Lower limb control
        self.torque_cmd =  action_scale* self.effort_limits[:, self.LB_joint_dof_ids] * self.actions
        # Apply torques
        self.robot.set_joint_effort_target(self.torque_cmd, joint_ids=self.LB_joint_dof_ids)


    def _compute_intermediate_values(self):
        # Joint data
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.joint_vel_scaled = self.joint_vel * self.joint_vel_scale
        
        # Bodies data
        self.body_pos_w = self.robot.data.body_pos_w
        self.body_quat_w = self.robot.data.body_quat_w
        self.body_lin_vel_w = self.robot.data.body_lin_vel_w
        self.body_ang_vel_w = self.robot.data.body_ang_vel_w

        # COM
        self.com_parts_w = self.robot.data.body_com_pos_w

        (
            self.joint_pos_scaled,
            self.torso_heading_proj,
            self.torso_up_proj,
            self.yaw_error,
            self.pelv_heading_proj,
            self.pelv_up_proj,
            self.l_foot_side_proj,
            self.r_foot_side_proj,
            self.phase,
            self.l_foot_target_pelv,
            self.r_foot_target_pelv,
            self.l_foot_rel_pos_pelv,
            self.r_foot_rel_pos_pelv,
            self.com_w,
            self.com_x_error,
            self.com_y_error,
            self.prev_potentials,
            self.potentials,
            self.L,
        ) = compute_intermediate_values(
            # joint control
            self.joint_pos,
            self.joint_limits_lower,
            self.joint_limits_upper,
            # bodies control
            self.inv_start_rot,
            self.body_pos_w,
            self.body_quat_w,
            self.body_lin_vel_w,
            self.body_ang_vel_w,
            self.pelvis_idx,
            self.torso_idx,
            self.l_foot_idx,
            self.r_foot_idx,
            self.UB_link_ids_tensor,
            # dynamics
            self.mass,
            self.com_parts_w,
            # targets
            self.targets_w,
            self.basis_heading_vec,
            self.basis_up_vec,
            self.basis_side_vec,
            self.potentials,
            self.prev_potentials,
            # foot steps
            self.gait_time_step,
            self.period,
            self.step_length,
            self.foot_z_offset,
            # sim timestep
            self.rl_dt,
            self.env_time_step,
        )

    def _get_observations(self) -> dict:
        # Time step control
        self.sim_time_step += 1
        self.env_time_step += 1
        self.gait_time_step += 1
        self.gait_time_step %= self.period
        # Update
        self.update_target()
        self._compute_intermediate_values()
        self.update_marker()
        obs = torch.cat(
            (
                self.joint_pos_scaled[:, self.LB_joint_dof_ids], # n actions
                self.joint_vel_scaled[:, self.LB_joint_dof_ids], # n actions
                self.actions, # n actions
                self.pelv_heading_proj.unsqueeze(-1), # 1
                self.pelv_up_proj.unsqueeze(-1), # 1 
                self.body_quat_w[:,self.pelvis_idx], # 4
                self.phase.unsqueeze(-1), # 1
                self.targets_rel_w, # 3
            ),
            dim=-1
        )
        return {"policy": obs}
    
    
    def _get_rewards(self) -> torch.Tensor:
        total_reward, reward_terms = compute_rewards(
                self.joint_pos_scaled[:, self.LB_joint_dof_ids],
                self.joint_vel_scaled[:, self.LB_joint_dof_ids],
                self.actions,
                self.torso_heading_proj,
                self.torso_up_proj,
                self.yaw_error,
                self.pelv_heading_proj,
                self.pelv_up_proj,
                self.l_foot_side_proj,
                self.r_foot_side_proj,
                self.l_foot_target_pelv,
                self.r_foot_target_pelv,
                self.l_foot_rel_pos_pelv,
                self.r_foot_rel_pos_pelv,
                self.com_x_error,
                self.com_y_error,
                self.prev_potentials,
                self.potentials,
                self.target_vel,
                self.body_lin_vel_w[:, self.pelvis_idx],
                self.L,
                self.reset_terminated,
            )
        
        self.reward_terms = reward_terms
        if int(self.sim_time_step % 100) == 0:
            self.save_reward_log()
            
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
        self.targets_w[env_ids] = self.init_targets_w[env_ids]
        self.potentials[env_ids] = 0
        self.prev_potentials[env_ids] = 0

        to_target = self.targets_w[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0

        self.env_time_step[env_ids] = 0
        self.gait_time_step[env_ids] = 0

    def save_reward_log(self):
        if not hasattr(self, "reward_terms"):
            return

        means = self.reward_terms.mean(dim=0).detach().cpu().numpy()

        step = int(self.sim_time_step)

        with open(self.log_path, "a") as f:
            f.write(f"{step}," + ",".join([f"{v:.6f}" for v in means]) + "\n")



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
    pelvis_ids:int,
    torso_idx:int,
    l_foot_idx:int,
    r_foot_idx:int,
    UB_link_ids_tensor:torch.Tensor,
    mass:torch.Tensor,
    com_parts_w:torch.Tensor,
    targets_w:torch.Tensor,
    basis_heading_vec:torch.Tensor,
    basis_up_vec:torch.Tensor,
    basis_side_vec:torch.Tensor,
    potentials:torch.Tensor,
    prev_potentials:torch.Tensor,
    gait_time_step:torch.Tensor,
    period:float,
    step_length:float,
    foot_z_offset:torch.Tensor,
    rl_dt:float,
    time_step:torch.Tensor,
):
    ### JOINT POSITION SCALE ###
    joint_pos_scaled = 2.0 * (joint_pos - joint_limits_lower) / (joint_limits_upper - joint_limits_lower) - 1.0

    ### TORSO CALCULATION ###
    # Torso pose [world frame]
    torso_pos_w = body_pos_w[:, torso_idx]
    torso_quat_w = body_quat_w[:, torso_idx]

    # To target [world frame]
    to_target = targets_w - torso_pos_w
    to_target[:, 2] = 0.0
    to_target_dir = normalize(to_target)
    
    # Direction vectors  [world frame]
    torso_heading_vec_w = quat_rotate(torso_quat_w, basis_heading_vec) # rotate basis heading vector
    torso_up_vec_w = quat_rotate(torso_quat_w, basis_up_vec) # rotate basis up vector
    
    # Heading alignment with target direciton
    torso_heading_proj = torch.sum(torso_heading_vec_w * to_target_dir, dim=-1)
    
    # Up projection in world z-direction
    torso_up_proj = torso_up_vec_w[:, 2]
    
    # Roll, Pitch, Yaw
    # roll = torch.atan2(torso_up_vec_w[:, 1], torso_up_vec_w[:, 2])
    # pitch = torch.atan2(-torso_up_vec_w[:, 0], torso_up_vec_w[:, 2])
    yaw = torch.atan2(torso_heading_vec_w[:, 1], torso_heading_vec_w[:, 0])
    
    # Heading error
    yaw_target = torch.atan2(to_target_dir[:, 1], to_target_dir[:, 0])
    yaw_error = yaw_target - yaw
    yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))

    ### PELVIS BODY CALCULATION ###
    # Upper body rotation [world frame]
    pelv_pos_w = body_pos_w[:, pelvis_ids]
    pelv_quat_w = body_quat_w[:, pelvis_ids]

    # Direction vectors  [world frame]
    pelv_heading_vec_w = quat_rotate(pelv_quat_w, basis_heading_vec) # rotate basis heading vector
    pelv_up_vec_w = quat_rotate(pelv_quat_w, basis_up_vec) # rotate basis up vector

    # Heading alignment with target direciton
    pelv_heading_proj = torch.sum(pelv_heading_vec_w * to_target_dir, dim=-1)
    
    # Up projection in world z-direction
    pelv_up_proj = pelv_up_vec_w[:, 2]

    ### FOOT CALCULATION ###
    # Pelvis canonical frame
    pelv_quat_can = quat_mul(inv_start_rot, pelv_quat_w)

    # Foot data [world frame]
    l_foot_pos_w = body_pos_w[:, l_foot_idx]
    l_foot_quat_w = body_quat_w[:, l_foot_idx]
    r_foot_pos_w = body_pos_w[:, r_foot_idx]
    r_foot_quat_w = body_quat_w[:, r_foot_idx]

    # Foot positions [pelvis canonial frame]
    l_foot_rel_pos_w = l_foot_pos_w - pelv_pos_w
    l_foot_rel_pos_pelv = quat_rotate_inverse(pelv_quat_can, l_foot_rel_pos_w)
    r_foot_rel_pos_w = r_foot_pos_w - pelv_pos_w
    r_foot_rel_pos_pelv = quat_rotate_inverse(pelv_quat_can, r_foot_rel_pos_w)

    # Foot directions [world frame]
    l_foot_side_vec_w = quat_rotate(l_foot_quat_w, basis_side_vec)
    r_foot_side_vec_w = quat_rotate(r_foot_quat_w, basis_side_vec)

    # Foot vectors projection
    pelv_side_vec_w = quat_rotate(pelv_quat_w, basis_side_vec)
    l_foot_side_proj = torch.sum(l_foot_side_vec_w * pelv_side_vec_w, dim=-1)
    r_foot_side_proj = torch.sum(r_foot_side_vec_w * pelv_side_vec_w, dim=-1)

    ### GAIT CONTROL ###
    # Gait phase
    phase = 2.0 * torch.pi * (gait_time_step.float() / period) # 0 ~ 2*pi
    left_swing_mask = torch.sin(phase) > 0.0 # l swing for 0 ~ pi
    right_swing_mask = torch.sin(phase) < 0.0 # r swing for pi ~ 2*pi
    left_stance_mask = ~left_swing_mask
    right_stance_mask = ~right_swing_mask

    # Stance position
    foot_y_offset = 0.1
    step_height = 0.05 # 15?
    stance_x = 0.0
    stance_z = foot_z_offset + 0.01 + step_height / 2

    # Desired trajectories
    l_foot_target_pelv = torch.zeros_like(l_foot_rel_pos_pelv)
    r_foot_target_pelv = torch.zeros_like(r_foot_rel_pos_pelv)

    # X direction
    l_foot_target_pelv[:, 0] = stance_x - step_length * torch.cos(phase)
    r_foot_target_pelv[:, 0] = stance_x + step_length * torch.cos(phase)
    init_mask = time_step < int(period/4) # for the first pi/2 period
    l_foot_target_pelv[init_mask, 0] = 0
    r_foot_target_pelv[init_mask, 0] = 0
    
    # Y direction
    l_foot_target_pelv[:, 1] = foot_y_offset
    r_foot_target_pelv[:, 1] = -foot_y_offset

    # Z direction
    l_foot_target_pelv[:, 2] = stance_z + step_height * torch.sin(phase) * left_swing_mask.float()
    r_foot_target_pelv[:, 2] = stance_z - step_height * torch.sin(phase) * right_swing_mask.float()
    l_foot_target_pelv[left_stance_mask, 2] = l_foot_rel_pos_pelv[left_stance_mask, 2] - 0.01
    r_foot_target_pelv[right_stance_mask, 2] = r_foot_rel_pos_pelv[right_stance_mask, 2] - 0.01


    ### COM CONTROL ###
    # Calculate COM
    mass = mass.unsqueeze(-1)
    # weighted_pos = com_parts_w * mass
    # com_w = weighted_pos.sum(dim=1) / mass.sum(dim=1)
    # com_rel = com_w - pelv_pos_w
    # COM for upper body
    mass_ub = mass[:, UB_link_ids_tensor]
    weighted_pos = com_parts_w[:, UB_link_ids_tensor] * mass_ub
    com_w = weighted_pos.sum(dim=1) / mass_ub.sum(dim=1)
    com_rel = com_w - pelv_pos_w
    # COM target
    com_x_target = 0.03
    com_x_error = com_rel[:, 0] - com_x_target
    # com_y_target = (l_foot_rel_pos_pelv[left_stance_mask, 1] + r_foot_rel_pos_pelv[right_stance_mask, 1])/2
    # com_y_target = torch.where(
    #     left_stance_mask,
    #     l_foot_rel_pos_pelv[:, 1],
    #     r_foot_rel_pos_pelv[:, 1]
    # )
    com_y_target = 0.0
    com_y_error = com_rel[:, 1] - com_y_target

    ### POTENTIAL TO TARGET ### 
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / rl_dt

    ### ANGULAR MOMENTUM (Simple) ###
    r = body_pos_w - com_w.unsqueeze(1)
    p = mass * body_lin_vel_w
    L = torch.sum(torch.cross(r, p, dim=-1), dim=1)

    return (
        joint_pos_scaled,
        torso_heading_proj,
        torso_up_proj,
        yaw_error,
        pelv_heading_proj,
        pelv_up_proj,
        l_foot_side_proj,
        r_foot_side_proj,
        phase,
        l_foot_target_pelv,
        r_foot_target_pelv,
        l_foot_rel_pos_pelv,
        r_foot_rel_pos_pelv,
        com_w,
        com_x_error,
        com_y_error,
        prev_potentials,
        potentials,
        L,
    )

@torch.jit.script
def compute_rewards(
    joint_pos_scaled:torch.Tensor,
    joint_vel_scaled:torch.Tensor,
    actions:torch.Tensor,
    torso_heading_proj:torch.Tensor,
    torso_up_proj:torch.Tensor,
    yaw_error:torch.Tensor,
    pelv_heading_proj:torch.Tensor,
    pelv_up_proj:torch.Tensor,
    l_foot_side_proj:torch.Tensor,
    r_foot_side_proj:torch.Tensor,
    l_foot_target_pelv:torch.Tensor,
    r_foot_target_pelv:torch.Tensor,
    l_foot_rel_pos_pelv:torch.Tensor,
    r_foot_rel_pos_pelv:torch.Tensor,
    com_x_error:torch.Tensor,
    com_y_error:torch.Tensor,
    prev_potentials:torch.Tensor,
    potentials:torch.Tensor,
    target_vel:float,
    pelv_lin_vel_w:torch.Tensor,
    L:torch.Tensor,
    reset_terminated:torch.Tensor,
):
    # weights
    weight_joint_limit = 0.15
    weight_action      = 0.005
    weight_energy      = 0.003
    weight_pelv_vel_x  = 1.2
    weight_yaw         = 0.2
    weight_torso_head  = 0.35
    weight_torso_up    = 0.45
    weight_pelv_head   = 0.35
    weight_pelv_up     = 0.45
    weight_lfoot_side  = 0.10
    weight_rfoot_side  = 0.10
    weight_lfoot_x     = 0.25
    weight_lfoot_y     = 0.10
    weight_lfoot_z     = 0.20
    weight_rfoot_x     = 0.25
    weight_rfoot_y     = 0.10
    weight_rfoot_z     = 0.20
    weight_alive       = 0.05
    weight_progress    = 0.6
    weight_com         = 0.20
    weight_L           = 1e-5
    weight_death       = 5.0

    # [Penalty] Soft joint limits
    limit_margin = 0.8
    dof_limit_violation = torch.relu(torch.abs(joint_pos_scaled) - limit_margin)
    p_joint_limits = torch.sum(dof_limit_violation, dim=-1) * weight_joint_limit

    # [Penalty] Action
    p_actions = torch.mean(actions**2, dim=-1) * weight_action

    # [Penalty] Energy
    p_energy = torch.mean(torch.abs(actions * joint_vel_scaled), dim=-1) * weight_energy

    # [Reward] Pelv velocity
    vel_x_error = target_vel - pelv_lin_vel_w[:, 0]
    r_pelv_vel_x = torch.exp(-20.0 * torch.square(vel_x_error)) * weight_pelv_vel_x

    # [Reward] Yaw (head angle)
    r_yaw = torch.exp(-20.0 * torch.square(yaw_error)) * weight_yaw

    # [Reward] Torso head
    dir_thres = 0.9
    reward_ones = torch.ones_like(torso_heading_proj)
    r_torso_head = torch.where(torso_heading_proj > dir_thres, 
                               reward_ones, 
                               torch.clamp(torso_heading_proj / dir_thres, min=0.0)) * weight_torso_head
    # [Reward] Torso up
    r_torso_up = torch.where(torso_up_proj > dir_thres, 
                               reward_ones, 
                               torch.clamp(torso_up_proj / dir_thres, min=0.0)) * weight_torso_up
    # [Reward] Pelvis head
    r_pelv_head = torch.where(pelv_heading_proj > dir_thres, 
                               reward_ones, 
                               torch.clamp(pelv_heading_proj / dir_thres, min=0.0)) * weight_pelv_head
    # [Reward] Pelvis up
    r_pelv_up = torch.where(pelv_up_proj > dir_thres, 
                               reward_ones, 
                               torch.clamp(pelv_up_proj / dir_thres, min=0.0)) * weight_pelv_up
    # [Reward] Left foot side
    r_lfoot_side = torch.where(l_foot_side_proj > dir_thres, 
                               reward_ones, 
                               torch.clamp(l_foot_side_proj / dir_thres, min=0.0)) * weight_lfoot_side
    # [Reward] Right foot side
    r_rfoot_side = torch.where(r_foot_side_proj > dir_thres, 
                               reward_ones, 
                               torch.clamp(r_foot_side_proj / dir_thres, min=0.0)) * weight_rfoot_side
    # [Reward] Foot traj error
    l_x_error = l_foot_target_pelv[:, 0] - l_foot_rel_pos_pelv[:, 0]
    l_y_error = l_foot_target_pelv[:, 1] - l_foot_rel_pos_pelv[:, 1]
    l_z_error = l_foot_target_pelv[:, 2] - l_foot_rel_pos_pelv[:, 2]
    r_x_error = r_foot_target_pelv[:, 0] - r_foot_rel_pos_pelv[:, 0]
    r_y_error = r_foot_target_pelv[:, 1] - r_foot_rel_pos_pelv[:, 1]
    r_z_error = r_foot_target_pelv[:, 2] - r_foot_rel_pos_pelv[:, 2]
    r_lfoot_x = torch.exp(-5.0 * torch.square(l_x_error)) * weight_lfoot_x
    r_lfoot_y = torch.exp(-5.0 * torch.square(l_y_error)) * weight_lfoot_y
    r_lfoot_z = torch.exp(-5.0 * torch.square(l_z_error)) * weight_lfoot_z
    r_rfoot_x = torch.exp(-5.0 * torch.square(r_x_error)) * weight_rfoot_x
    r_rfoot_y = torch.exp(-5.0 * torch.square(r_y_error)) * weight_rfoot_y
    r_rfoot_z = torch.exp(-5.0 * torch.square(r_z_error)) * weight_rfoot_z
    r_foot_track = r_lfoot_x + r_lfoot_y + r_lfoot_z + r_rfoot_x + r_rfoot_y + r_rfoot_z

    # [Reward] Alive
    r_alive = torch.ones_like(potentials) * weight_alive

    # [Reward] progress
    r_progress = (potentials - prev_potentials) * weight_progress

    # [Reward] COM
    r_com_x = torch.exp(-5.0 * torch.square(com_x_error)) * weight_com
    
    # [Penalty] Angular momentum
    p_L = torch.tanh(torch.abs(L[:, 2] / 20.0)) * weight_L

    total_reward = (
        -p_joint_limits
        -p_actions
        -p_energy
        +r_pelv_vel_x
        +r_yaw
        +r_torso_head
        +r_torso_up
        +r_pelv_head
        +r_pelv_up
        +r_lfoot_side
        +r_rfoot_side
        +r_foot_track
        +r_alive
        +r_progress
        +r_com_x
        -p_L
    )

    death_cost = -weight_death
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)

    reward_terms = torch.stack([
        p_joint_limits,
        p_actions,
        p_energy,
        r_pelv_vel_x,
        r_yaw,
        r_torso_head,
        r_torso_up,
        r_pelv_head,
        r_pelv_up,
        r_lfoot_side,
        r_rfoot_side,
        r_lfoot_x,
        r_lfoot_y,
        r_lfoot_z,
        r_rfoot_x,
        r_rfoot_y,
        r_rfoot_z,
        r_alive,
        r_progress,
        r_com_x,
        p_L,
        total_reward,
    ], dim=-1)

    return total_reward, reward_terms
    