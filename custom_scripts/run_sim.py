"""
Docstring for run_sim
"""

####################################
# Part 1. Ready for the simulation #
####################################
import argparse
from isaaclab.app import AppLauncher

# Arguments parser
parser = argparse.ArgumentParser(description="MSK Simulation for Human/Humanoid/Exoskeleton control.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# import pacakages
import torch, math
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg, OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

####################################
# Part 2. Set Scene Configurations #
####################################
from msk_isaac.robots.humanbody import HUMANBODY_CFG, HumanbodyEnv, HumanbodyEnvCfg # import robot configuration
from msk_isaac.custom_math.utils import getRobotDynamicProperties, getRobotInformation
from msk_isaac.custom_math.iksolver import DiffIKSolver, OSCIKSolver
from msk_isaac.custom_math.quaternion import *

# Robot actuator parameters setup
for _, actuator_cfg in HUMANBODY_CFG.actuators.items():
    actuator_cfg.stiffness = 0.0
    actuator_cfg.damping = 0.0

# Scene
@configclass
class SceneCfg(InteractiveSceneCfg):
    
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot = HUMANBODY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

# Simulation
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    
    # get robot from scene entities
    robot = scene["robot"]

    print(getRobotInformation(robot))

    joint_limits = robot.data.joint_pos_limits
    joint_lower = joint_limits[0,:,0]
    joint_upper = joint_limits[0,:,1]
    print(joint_lower, joint_upper)

    effort_limits = robot.data.joint_effort_limits
    print(effort_limits)

    exit()

    # Obtain indices for the end-effector and arm joints
    l_arm_joint_ids, _ = robot.find_joints(["left_shoulder_.*", "left_elbow_.*"])
    r_arm_joint_ids, _ = robot.find_joints(["right_shoulder_.*", "right_elbow_.*"])
    l_leg_joint_ids, _ = robot.find_joints(["left_thigh_.*", "left_knee_.*", "left_ankle_.*"])
    r_leg_joint_ids, _ = robot.find_joints(["right_thigh_.*", "right_knee_.*", "right_ankle_.*"])
    waist_joint_ids, _ = robot.find_joints(["waist_.*"])

    # define simulation stepping
    sim_dt = sim.get_physics_dt()

    # update existing buffers
    robot.update(dt=sim_dt)

    # zero commands
    zero_joint_efforts = torch.zeros(scene.num_envs, robot.num_joints, device=sim.device)

    # define robot chain entities
    l_arm_entity_cfg = SceneEntityCfg("robot", 
                                      joint_ids=l_arm_joint_ids, 
                                      body_names=["upperbody", "left_lowerarm"], 
                                      preserve_order=True)
    r_arm_entity_cfg = SceneEntityCfg("robot", 
                                      joint_ids=r_arm_joint_ids, 
                                      body_names=["upperbody", "right_lowerarm"], 
                                      preserve_order=True)
    l_leg_entity_cfg = SceneEntityCfg("robot", 
                                      joint_ids=l_leg_joint_ids, 
                                      body_names=["upperbody", "left_foot"], 
                                      preserve_order=True)
    r_leg_entity_cfg = SceneEntityCfg("robot", 
                                      joint_ids=r_leg_joint_ids, 
                                      body_names=["base_link", "right_foot"], 
                                      preserve_order=True)
    waist_entity_cfg = SceneEntityCfg("robot", 
                                      joint_ids=waist_joint_ids, 
                                      body_names=["base_link", "upperbody"], 
                                      preserve_order=True)
    
    l_arm_entity_cfg.resolve(scene)
    r_arm_entity_cfg.resolve(scene)
    l_leg_entity_cfg.resolve(scene)
    r_leg_entity_cfg.resolve(scene)
    waist_entity_cfg.resolve(scene)

    def print_robot_info():
        # print robot info.
        print("\n-------------------")
        print("Robot Description")
        print("-------------------")
        print("[INFO]: joint num: ", robot.num_joints)
        print("[INFO]: link names")
        body_ids, body_names = robot.find_bodies(robot.body_names)
        _temp_str = ""
        for link_idx, link_name in zip(body_ids, body_names):
            _temp_str += f"{link_name}({link_idx}), "
        print(_temp_str)
        print("[INFO]: joint names: ")
        joint_ids, joint_names = robot.find_joints(robot.joint_names)
        _temp_str = ""
        for joint_idx, joint_name in zip(joint_ids, joint_names):
            _temp_str += f"{joint_name}({joint_idx}), "
        print(_temp_str)
        print("\n---------------------------")
        print("ROBOT JOINT CONFIGURATION")
        print("---------------------------")
        print("WAIST")
        for j_name, j_idx in zip(waist_entity_cfg.joint_names, waist_entity_cfg.joint_ids):
            print("|-", j_name, "(index: ", j_idx, ")")
        print("|-LEFT ARM")
        for j_name, j_idx in zip(l_arm_entity_cfg.joint_names, l_arm_entity_cfg.joint_ids):
            print("  |-", j_name, "(index: ", j_idx, ")")
        print("|-RIGHT ARM")
        for j_name, j_idx in zip(r_arm_entity_cfg.joint_names, r_arm_entity_cfg.joint_ids):
            print("  |-", j_name, "(index: ", j_idx, ")")
        print("|-LEFT LEG")
        for j_name, j_idx in zip(l_leg_entity_cfg.joint_names, l_leg_entity_cfg.joint_ids):
            print("  |-", j_name, "(index: ", j_idx, ")")
        print("|-RIGHT LEG")
        for j_name, j_idx in zip(r_leg_entity_cfg.joint_names, r_leg_entity_cfg.joint_ids):
            print("  |-", j_name, "(index: ", j_idx, ")")
        print("\n")
    print_robot_info()

    # Differential IK solver
    # ik_solver_l_arm = DiffIKSolver("L_Arm", sim, scene, l_arm_entity_cfg, command_type="pose", ik_method="pinv")
    # ik_solver_r_arm = DiffIKSolver("R_Arm", sim, scene, r_arm_entity_cfg, command_type="pose", ik_method="pinv")
    # ik_solver_l_leg = DiffIKSolver("L_Leg", sim, scene, l_leg_entity_cfg, command_type="pose", ik_method="pinv")
    # ik_solver_r_leg = DiffIKSolver("R_Leg", sim, scene, r_leg_entity_cfg, command_type="pose", ik_method="pinv")
    # ik_solver_waist = DiffIKSolver("Waist", sim, scene, waist_entity_cfg, command_type="pose", ik_method="pinv")

    # # Operational Space IK solver
    ik_solver_l_arm = OSCIKSolver("L_Arm", sim, scene, l_arm_entity_cfg, 
                                  target_types=["pose_abs"], 
                                  impedance_mode="fixed")
    ik_solver_r_arm = OSCIKSolver("R_Arm", sim, scene, r_arm_entity_cfg, 
                                  target_types=["pose_abs"], 
                                  impedance_mode="fixed")
    ik_solver_l_leg = OSCIKSolver("L_Leg", sim, scene, l_leg_entity_cfg, 
                                  target_types=["pose_abs"], 
                                  impedance_mode="fixed")
    ik_solver_r_leg = OSCIKSolver("R_Leg", sim, scene, r_leg_entity_cfg, 
                                  target_types=["pose_abs"], 
                                  impedance_mode="fixed")
    ik_solver_waist = OSCIKSolver("Waist", sim, scene, waist_entity_cfg, 
                                  target_types=["pose_abs"], 
                                  impedance_mode="fixed")

    target_l_arm_pose = ik_solver_l_arm.init_ee_pose_b.clone()
    target_r_arm_pose = ik_solver_r_arm.init_ee_pose_b.clone()
    target_l_leg_pose = ik_solver_l_leg.init_ee_pose_b.clone()
    target_r_leg_pose = ik_solver_r_leg.init_ee_pose_b.clone()

    count = 0
    while simulation_app.is_running():
        # reset simulation
        if count % 500 == 0: # every 5 seconds
            # reset joint state to default
            default_joint_pos = robot.data.default_joint_pos.clone()
            default_joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
            robot.set_joint_effort_target(zero_joint_efforts)
            robot.write_data_to_sim()
            robot.reset()

            ik_solver_l_arm.reset()
            ik_solver_r_arm.reset()
            ik_solver_l_leg.reset()
            ik_solver_r_leg.reset()
            ik_solver_waist.reset()
            
            target_l_leg_pose[:,1] -= 0.05
            target_l_leg_pose[:,2] += 0.05
            ik_solver_l_arm.set_command(target_l_arm_pose)
            ik_solver_r_arm.set_command(target_r_arm_pose)
            ik_solver_l_leg.set_command(target_l_leg_pose)
            ik_solver_r_leg.set_command(target_r_leg_pose)

        else:
            print("=== root ===")
            print(robot.data.root_pos_w[0])
            print(robot.data.root_quat_w[0])

            # apply actions (Diff)
            # l_arm_joint_pos_des = ik_solver_l_arm.compute()
            # robot.set_joint_position_target(l_arm_joint_pos_des, joint_ids=l_arm_joint_ids)
            # r_arm_joint_pos_des = ik_solver_r_arm.compute()
            # robot.set_joint_position_target(r_arm_joint_pos_des, joint_ids=r_arm_joint_ids)
            # l_leg_joint_pos_des = ik_solver_l_leg.compute()
            # robot.set_joint_position_target(l_leg_joint_pos_des, joint_ids=l_leg_joint_ids)
            # r_leg_joint_pos_des = ik_solver_r_leg.compute()
            # robot.set_joint_position_target(r_leg_joint_pos_des, joint_ids=r_leg_joint_ids)
            # waist_joint_pos_des = ik_solver_waist.compute()
            # robot.set_joint_position_target(waist_joint_pos_des, joint_ids=waist_joint_ids)
            
            # # apply actions (OSC)
            # l_arm_joint_effort = ik_solver_l_arm.compute()
            # robot.set_joint_effort_target(l_arm_joint_effort, joint_ids=l_arm_joint_ids)
            # r_arm_joint_effort = ik_solver_r_arm.compute()
            # robot.set_joint_effort_target(r_arm_joint_effort, joint_ids=r_arm_joint_ids)
            # l_leg_joint_effort = ik_solver_l_leg.compute()
            # robot.set_joint_effort_target(l_leg_joint_effort, joint_ids=l_leg_joint_ids)
            # r_leg_joint_effort = ik_solver_r_leg.compute()
            # robot.set_joint_effort_target(r_leg_joint_effort, joint_ids=r_leg_joint_ids)
            # waist_joint_effort = ik_solver_waist.compute()
            # robot.set_joint_effort_target(waist_joint_effort, joint_ids=waist_joint_ids)

            # robot.write_data_to_sim()
            

        # perform step
        sim.step(render=True)
        # update robot buffers
        robot.update(dt=sim_dt)
        # update scene
        scene.update(dt=sim_dt)
        # update sim-time
        count += 1


def main():
    # Load sim
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view([3.5, 3.5, 3.5], [0.5, 0.5, 0.5])

    # Design scene
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()


