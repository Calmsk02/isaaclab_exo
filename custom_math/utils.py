import torch, time
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

def getRobotDynamicProperties(
            robot: Articulation,
            base_link_names: list,
            ee_link_names: list,
            joint_names: list):
    
    t0 = time.time()
    torch.set_printoptions(precision=3)
    joint_ids, joint_names = robot.find_joints(joint_names)
    base_link_ids, base_link_names = robot.find_bodies(base_link_names)
    ee_link_ids, ee_link_names  = robot.find_bodies(ee_link_names)

    # Poses for link frames
    # (num_envs, num_ee, pose(7))
    n_branches = len(ee_link_ids)
    base_pose_w = robot.data.body_pose_w[:, base_link_ids].repeat(1, n_branches, 1)
    ee_pose_w = robot.data.body_pose_w[:, ee_link_ids]
    # change to base frame
    base_pose_w_flat = base_pose_w.reshape(-1, 7)
    ee_pose_w_flat = ee_pose_w.reshape(-1, 7)
    ee_pos_b_flat, ee_quat_b_flat = subtract_frame_transforms(
        base_pose_w_flat[:, :3],
        base_pose_w_flat[:, 3:7],
        ee_pose_w_flat[:, :3],
        ee_pose_w_flat[:, 3:7]
    )
    ee_pose_b_flat = torch.cat([ee_pos_b_flat, ee_quat_b_flat], dim=-1)
    ee_pose_b = ee_pose_b_flat.reshape(-1, n_branches, 7)
    
    # Task De-coupled Jacobian
    # Fixed base: (num_envs, num_links-1, 6, num_joints)
    # Float base: (num_envs, num_links, 6, base_pose(6)+num_joints)
    ee_jacobi_ids = []
    if robot.is_fixed_base:
        jacobian_w = robot.root_physx_view.get_jacobians()
        for idx in ee_link_ids:
            ee_jacobi_ids.append(idx - 1)
    else:
        jacobian_w = robot.root_physx_view.get_jacobians()[..., 6:]
        ee_jacobi_ids = ee_link_ids

    ee_jacobian_w = jacobian_w[:, ee_jacobi_ids, :, :]

    # COM calculation
    # get_masses() returns tensor in cpu
    # J_com_w: (num_envs, 1, 3, num_joints)
    if robot.is_fixed_base:
        mass = robot.root_physx_view.get_masses()[:,1:].to(jacobian_w.device)
        com_w = robot.data.body_com_pos_w[:,1:,:]
    else:
        mass = robot.root_physx_view.get_masses().to(jacobian_w.device)
        com_w = robot.data.body_com_pos_w
    total_mass = mass.sum(dim=1, keepdim=True)
    com_jacobian_v_w = jacobian_w[:,:,:3,:] * mass.unsqueeze(-1).unsqueeze(-1)
    com_jacobian_v_w = com_jacobian_v_w.sum(dim=1, keepdim=True) / total_mass

    # Jacobian calculation
    # return:  [[ jv_com_w ]
    #           [ j_ee_w ]]
    # (num_envs, task_dimensions, num_joint)
    Jacobian_w = torch.cat([com_jacobian_v_w.reshape(jacobian_w.shape[0], -1, jacobian_w.shape[-1]), 
                            ee_jacobian_w.reshape(jacobian_w.shape[0], -1, jacobian_w.shape[-1])], 
                            dim=1)
    
    
    J_inv_w = torch.linalg.pinv(Jacobian_w)
    
    t1 = time.time()

def getRobotInformation(robot: Articulation):

    print("---------------------")
    print("- Robot Inforamtion -")
    print("---------------------")
    ids, names = robot.find_joints([".*"])
    for idx, name in zip(ids, names):
        print(idx, name)
    ids, names = robot.find_bodies([".*"])
    for idx, name in zip(ids, names):
        print(idx, name)
    print("\n")
