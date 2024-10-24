import rclpy
from rclpy.node import Node
from unitree_ros2_real import UnitreeRos2Real, get_euler_xyz

import os
import os.path as osp
import json
import time
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# from rsl_rl import modules

from logger import Logger
from models import ZeroActModel, StandUpModel, SitDownModel, ActorModel, Recovery_Controller, YangTraj, FuTraj


class Go2Node(UnitreeRos2Real):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_class_name= "Go2", **kwargs)
        
        self.data_logger = Logger()
        self.log_cnt = 0
        self.stop_log_cnt = 50

    def register_models(self, 
                        stand_policy, 
                        sit_policy, 
                        recovery_policy, 
                        animal_policy,
                        beast_policy,
                        yang_policy,
                        fu_policy,
                        ):
        
        self.stand_policy = stand_policy
        self.sit_policy = sit_policy
        self.recovery_policy = recovery_policy

        self.animal_policy = animal_policy
        self.beast_policy = beast_policy
        
        self.yang_policy = yang_policy
        self.fu_policy = fu_policy
               
        self.base_projected_G = torch.zeros((1, 3), device= self.model_device, dtype= torch.float32)
        self.base_projected_G[:, 2] = -1

        self.FSM_STATE = 0
        # 0 : SIT
        # 1 : STAND
        # 2 : ANIMAL
        # 3 : BEAST
        # 4 : RECOVERY
        # 5 : Yang
        # 6 : Fu

    def start_main_loop_timer(self, duration):
        self.main_loop_timer = self.create_timer(
            duration, # in sec
            self.main_loop,
        )
        
    def main_loop(self):            

        if (self.joy_stick_buffer.keys & self.WirelessButtons.start): #! animal
            self.FSM_STATE = 2
        elif (self.joy_stick_buffer.keys & self.WirelessButtons.select): #! beast
            self.FSM_STATE = 3
        elif (self.joy_stick_buffer.keys & self.WirelessButtons.Y): #! sit
            self.FSM_STATE = 0
            self.sit_policy.reset()
        elif (self.joy_stick_buffer.keys & self.WirelessButtons.X): #! stand
            self.FSM_STATE = 1
            self.stand_policy.reset()
        elif (self.joy_stick_buffer.keys & self.WirelessButtons.A): #! recovery
            self.FSM_STATE = 4
            self.recovery_policy.reset()
        elif (self.joy_stick_buffer.keys & self.WirelessButtons.up): #! yang
            self.FSM_STATE = 5
            self.yang_policy.reset()
        elif (self.joy_stick_buffer.keys & self.WirelessButtons.down): #! fu
            self.FSM_STATE = 6
            self.fu_policy.reset()
            
        if (self.base_projected_G[0,2] > -0.1):
            self.FSM_STATE = 4
            self.recovery_policy.reset()

        if self.FSM_STATE == 2: #! animal

            self.base_projected_G = self._get_projected_gravity_obs()
            task_obs = self.get_obs()
            props_history = self.get_props_history()
            
            action = self.animal_policy(task_obs, props_history)
            
            self.post_physics_step()
             
            self.send_action(action)

        elif self.FSM_STATE == 3:   #! beast

            self.base_projected_G = self._get_projected_gravity_obs()
            task_obs = self.get_obs()
            props_history = self.get_props_history()

            action = self.beast_policy(task_obs, props_history)
            
            self.post_physics_step()
             
            self.send_action(action)
        
        elif self.FSM_STATE == 4:   #! recovery
            base_ang_vel_obs = self._get_ang_vel_obs()
            self.base_projected_G = self._get_projected_gravity_obs()
            dof_pos_obs = self._get_dof_pos_obs()
            dof_vel_obs = self._get_dof_vel_obs()
            action_obs = self._get_actions_obs()
            recovery_obs = torch.cat((base_ang_vel_obs, self.base_projected_G, dof_pos_obs, dof_vel_obs, action_obs), dim=1)
            
            rec_action_now = self.recovery_policy(recovery_obs)
            self.send_action(rec_action_now)

        elif self.FSM_STATE == 0:   #! sit
            obs = self._get_dof_pos() # do not multiply by obs_scales["dof_pos"]
            action = self.sit_policy(obs)
            self.send_action(action / self.action_scale)

        elif self.FSM_STATE == 1:   #! stand
            self.base_projected_G = self._get_projected_gravity_obs()
            obs = self._get_dof_pos() # do not multiply by obs_scales["dof_pos"]
            action = self.stand_policy(obs)
            self.send_action(action / self.action_scale)

        elif self.FSM_STATE == 5:   #! 
            obs = self._get_dof_pos() # do not multiply by obs_scales["dof_pos"]
            action = self.yang_policy(obs)
            self.send_action(action / self.action_scale)

        elif self.FSM_STATE == 6:   #! 
            obs = self._get_dof_pos() # do not multiply by obs_scales["dof_pos"]
            action = self.fu_policy(obs)
            self.send_action(action / self.action_scale)

        else:
            self.get_logger().info("No policy is selected")
            self.send_action(self._get_dof_pos_obs() / self.action_scale)


@torch.inference_mode()
def main(args):
    rclpy.init()

    #! load config
    # animal 
    assert args.logdir_animal_policy is not None, "Please provide a logdir"
    with open(osp.join(args.logdir_animal_policy, "env_cfg.json"), "r") as f:
        animal_env_config_dict = json.load(f, object_pairs_hook= OrderedDict)
    with open(osp.join(args.logdir_animal_policy, "train_cfg.json"), "r") as f:
        animal_alg_config_dict = json.load(f, object_pairs_hook= OrderedDict)
    # beast
    assert args.logdir_beast_policy is not None, "Please provide a logdir"
    with open(osp.join(args.logdir_beast_policy, "env_cfg.json"), "r") as f:
        beast_env_config_dict = json.load(f, object_pairs_hook= OrderedDict)
    with open(osp.join(args.logdir_beast_policy, "train_cfg.json"), "r") as f:
        beast_alg_config_dict = json.load(f, object_pairs_hook= OrderedDict)

    duration = animal_env_config_dict["sim"]["dt"] * animal_env_config_dict["control"]["decimation"] # in sec
    device = "cuda"  

    #! create hardware interfance
    env_node = Go2Node(
        "go2",
        cfg= animal_env_config_dict,
        replace_obs_with_embeddings= [],
        model_device= device,
        dryrun= not args.nodryrun,
    )

    # env_node.get_logger().info("Model loaded from: {}".format(osp.join(args.logdir_animal_policy, animal_model_names[-1])))
    env_node.get_logger().info("Control Duration: {} sec".format(duration))
    env_node.get_logger().info("Motor Stiffness (kp): {}".format(env_node.p_gains))
    env_node.get_logger().info("Motor Damping (kd): {}".format(env_node.d_gains))

    #! load the Animal model with the latest checkpoint
    animal_model_names = [i for i in os.listdir(args.logdir_animal_policy) if i.startswith("model_")]
    animal_model_names.sort(key= lambda x: int(x.split("_")[-1].split(".")[0]))
    animal_state_dict = torch.load(osp.join(args.logdir_animal_policy, animal_model_names[-1]), map_location= "cpu")

    #! load the Beast model with the latest checkpoint
    beast_model_names = [i for i in os.listdir(args.logdir_beast_policy) if i.startswith("model_")]
    beast_model_names.sort(key= lambda x: int(x.split("_")[-1].split(".")[0]))
    beast_state_dict = torch.load(osp.join(args.logdir_beast_policy, beast_model_names[-1]), map_location= "cpu")

    #! load models

    #* zero_act_model to start the safe standing
    # zero_act_model = ZeroActModel()
    # zero_act_model = torch.jit.script(zero_act_model)
    
    #* standup trajectory
    stand_model = StandUpModel()
    # stand_policy = torch.jit.script(stand_model)
    
    #* sitdown trajectory
    sit_model = SitDownModel()
    # sit_policy = torch.jit.script(sit_model)

    yang_model = YangTraj()
    fu_model = FuTraj()

    #* recovery controller
    recovery_model = Recovery_Controller(
        num_actor_obs = 42,
        num_critic_obs= 42,
        num_actions = 12,
        actor_hidden_dims = [512, 256, 128],
        critic_hidden_dims = [512, 256, 128]
    )
    model_names_recovery = [i for i in os.listdir(args.logdir_recovery) if i.startswith("model_")]
    model_names_recovery.sort(key= lambda x: int(x.split("_")[-1].split(".")[0]))
    state_dict_recovery = torch.load(osp.join(args.logdir_recovery, model_names_recovery[-1]), map_location= "cpu")
    recovery_model.load_state_dict(state_dict_recovery['model_state_dict'])
    recovery_model.eval()
    recovery_model.to(device)

    #* animal policy
    act_model_animal = ActorModel(        
        num_actor_obs = animal_env_config_dict["env"]["num_observations"],
        num_critic_obs = animal_env_config_dict["env"]["num_privileged_obs"],
        num_actions= animal_env_config_dict["env"]["num_actions"],
        history_step= 6,
        **animal_alg_config_dict["policy"])
    act_model_animal.load_state_dict(animal_state_dict["model_state_dict"])
    act_model_animal.eval()
    act_model_animal.to(device)

    @torch.jit.script
    def animal_policy(obs: torch.Tensor, props_history: torch.Tensor):
        action = act_model_animal(obs, props_history)
        return action
    
    #* beast policy
    act_model_beast = ActorModel(        
        num_actor_obs = animal_env_config_dict["env"]["num_observations"],
        num_critic_obs = animal_env_config_dict["env"]["num_privileged_obs"],
        num_actions= animal_env_config_dict["env"]["num_actions"],
        history_step= 6,
        **animal_alg_config_dict["policy"])
    act_model_beast.load_state_dict(beast_state_dict["model_state_dict"])
    act_model_beast.eval()
    act_model_beast.to(device)

    @torch.jit.script
    def beast_policy(obs: torch.Tensor, props_history: torch.Tensor):
        action = act_model_beast(obs, props_history)
        return action

    #! Init policy node
    env_node.register_models(

        stand_model,

        sit_model,

        recovery_model,
        
        animal_policy,
        
        beast_policy,
        
        yang_model,
        
        fu_model
    )

    #! Start hardware com nodes
    env_node.start_ros_handlers()

    if args.loop_mode == "while":
        rclpy.spin_once(env_node, timeout_sec= 0.)
        env_node.get_logger().info("Model and Policy are ready")
        while rclpy.ok():
            
            main_loop_time = time.monotonic()
            env_node.main_loop()
            
            env_node.get_logger().info("loop time: {:f}".format((time.monotonic() - main_loop_time)))
            
            spin_time = time.monotonic()
            rclpy.spin_once(env_node, timeout_sec= 0.)
           
            env_node.get_logger().info("spin time: {:f}".format((time.monotonic() - spin_time)))
            
            env_node.get_logger().info("sleep time: {:f}".format(max(0, duration - (time.monotonic() - main_loop_time))))
            time.sleep(max(0, duration - (time.monotonic() - main_loop_time)))
            
            env_node.get_logger().info("total time: {:f}".format((time.monotonic() - main_loop_time)))
            
    #! default timer
    elif args.loop_mode == "timer": 
        env_node.start_main_loop_timer(duration)
        rclpy.spin(env_node)
    rclpy.shutdown()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()  
    # weight logdir       
    parser.add_argument("--logdir_animal_policy", type= str, default= "/home/unitree/QTY_WS/legged_gym/logs/rough_go2/Sep16_16-25-28_9-16-R0.8+W0.4", help= "The directory which contains the config.json and model_*.pt files")
    parser.add_argument("--logdir_beast_policy", type= str, default= "/home/unitree/QTY_WS/legged_gym/logs/rough_go2/10-11-1", help= "The directory which contains the config.json and model_*.pt files")
    parser.add_argument("--logdir_recovery", type= str, default= "/home/unitree/QTY_WS/legged_gym/logs/go2_recovery/10-7-1", help= "The directory which contains the config.json and model_*.pt files")

    # enable motion
    parser.add_argument("--nodryrun", action= "store_true", default= True, help= "Disable dryrun mode")
    parser.add_argument("--loop_mode", type= str, default= "timer",
        choices= ["while", "timer"],
        help= "Select which mode to run the main policy control iteration",
    )

    args = parser.parse_args()
    main(args)
