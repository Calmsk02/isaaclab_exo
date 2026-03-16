"""
Docstring for robots.humanbody
"""
# Import isaaclab packages
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Get asset path
import os, math
ASSET_PATH = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'assets')

#######################
# Robot Configuration #
#######################
HUMANBODY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_PATH}/isaac_humanbody_description/urdf/isaac_humanbody_fixed_base/isaac_humanbody.usd",
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
