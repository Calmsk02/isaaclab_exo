Required: Isaaclab, python3  
How to use:  
- Place robot model urdf into the "source/msk_isaac/msk_isaac/assets/" folder.  
- Generate usd file in the isaacsim simulator.  
- Make your own robor environment in "source/msk_isaac/msk_isaac/robots/" folder.  
- Write additional code (controller, utilities .. ) in "source/msk_isaac/msk_isaac/custom_math/" folder.  
- Custom scripts (not using defined rl agents) are in "custom_scripts" folder.
  
Run commands:  
(requires alias in bash for using {isaaclab_path}/isaaclab.sh)  
{alias_cmd} -p scripts/{agent_name}/train.py --task={task_name} (--num_envs=) (--checkpoint=)  
{alias_cmd} -p scripts/{agent_name}/play.py --task={task_name} (--num_envs=) (--checkpoint=)  