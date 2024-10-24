import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ZeroActModel(torch.nn.Module):
    def __init__(self, angle_tolerance= 0.15, delta= 0.2):
        super().__init__()
        self.angle_tolerance = angle_tolerance
        self.delta = delta

    def forward(self, dof_pos): # 
        target = torch.zeros_like(dof_pos)
        diff = dof_pos - target
        diff_large_mask = torch.abs(diff) > self.angle_tolerance
        target[diff_large_mask] = dof_pos[diff_large_mask] - self.delta * torch.sign(diff[diff_large_mask])
        return target
    

class StandUpModel(torch.nn.Module):
    def __init__(self, stand_time=2.0):
        super().__init__()
        
        self.time_step = 0.02
        self.stand_time = stand_time
        #                                         FL                FR              RL              RR
        self.default_dof_pos = torch.tensor([0.1, 0.8, -1.5, 
                                             -0.1, 0.8, -1.5, 
                                             0.1, 1.0, -1.5, 
                                             -0.1, 1.0, -1.5], device="cuda:0", dtype= torch.float32)
        self.stand_dof_pos = torch.tensor([0.1, 0.8, -1.5, 
                                             -0.1, 0.8, -1.5, 
                                             0.1, 1.0, -1.5, 
                                             -0.1, 1.0, -1.5], device="cuda:0", dtype= torch.float32)
        self.init_motor_angles = torch.zeros_like(self.default_dof_pos, device="cuda:0")
        
        self.t = 0.0
        self.first = True

    def forward(self, relativa_dof_pos):
        if self.first:
            self.init_motor_angles = self.default_dof_pos + relativa_dof_pos
            self.first = False

        blend_ratio = min(self.t / self.stand_time, 1)
        self.t += self.time_step 
        
        action = (1 - blend_ratio)  * self.init_motor_angles + blend_ratio * self.stand_dof_pos 
        
        return action - self.default_dof_pos
    
    def reset(self):
        self.t = 0.0
        self.first = True


class SitDownModel(torch.nn.Module):
    def __init__(self, stand_time=3.0):
        super().__init__()
        
        self.time_step = 0.02
        self.stand_time = stand_time
        self.default_dof_pos = torch.tensor([0.1, 0.8, -1.5, 
                                             -0.1, 0.8, -1.5, 
                                             0.1, 1.0, -1.5, 
                                             -0.1, 1.0, -1.5], device="cuda:0", dtype= torch.float32)
        #                                         FL                FR              RL              RR
        self.sit_dof_pos = torch.tensor([0.0, 0.93, -2.6, 
                                             0.0, 0.93, -2.6, 
                                             0.0, 0.93, -2.6, 
                                             0.0, 0.93, -2.6], device="cuda:0", dtype= torch.float32)
        self.init_motor_angles = torch.zeros_like(self.default_dof_pos, device="cuda:0")
        
        self.t = 0.0
        self.first = True

    def forward(self, relativa_dof_pos):
        if self.first:
            self.init_motor_angles = self.default_dof_pos + relativa_dof_pos
            self.first = False

        blend_ratio = min(self.t / self.stand_time, 1)
        self.t += self.time_step 
        
        action = (1 - blend_ratio)  * self.init_motor_angles + blend_ratio * self.sit_dof_pos 
        
        return action - self.default_dof_pos
    
    def reset(self):
        self.t = 0.0
        self.first = True 

class Estimator(nn.Module):
    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 enc_hidden_dims=[128, 64, 16],
                 tar_hidden_dims=[64, 16],
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 num_prototype=32,
                 temperature=3.0,
                 #!
                 UseTCN=False,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
            
        super(Estimator, self).__init__()
        activation = torch.nn.ELU()

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.temperature = temperature
        self.UseTCN = UseTCN

        # Encoder
        enc_input_dim = self.temporal_steps * self.num_one_step_obs

        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]    #! hidden layers
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1]+3)]
        self.encoder_net = nn.Sequential(*enc_layers)

        tar_input_dim = 35
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target_net = nn.Sequential(*tar_layers)

        # Prototype
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])

    def forward(self, props_history):    
        props_history = props_history.view(-1, self.temporal_steps * self.num_one_step_obs)
        part = self.encoder_net(props_history.detach())
        latent1, latent2 = part[:, :3], part[:, 3:]
        latent2_normal = F.normalize(latent2, dim=-1, p=2)
        return latent1.detach(), latent2_normal.detach()

class ActorModel(torch.nn.Module):
    def __init__(self, 
                 num_actor_obs,
                 num_critic_obs,
                 num_actions,
                 history_step,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[256, 256, 256],
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorModel, self).__init__()
        
        activation = torch.nn.ELU()
        
        #! encoder
        self.estimator = Estimator(temporal_steps=history_step, num_one_step_obs=3*2 + 3*12)

        #! actor
        mlp_input_dim_a = num_actor_obs - 35 + 16 + 3
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)
        
        #! critic
        mlp_input_dim_c = num_critic_obs
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)
        
        self.std = nn.Parameter(0 * torch.ones(num_actions))
        
    def forward(self, observations, props_history):
        #todo
        vel, latent = self.estimator(props_history)
        actor_obs = torch.cat((observations[:,:-35], vel, latent), dim=-1)
        actions_mean = self.actor(actor_obs)
        return actions_mean
    
class Recovery_Controller(nn.Module):
    def __init__(self,  
                 num_actor_obs,
                num_critic_obs,
                num_actions,
                actor_hidden_dims=[256, 256, 256],
                critic_hidden_dims=[256, 256, 256],
                # activation='elu',
                # init_noise_std=1.0,
                # fixed_std=False,
                **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(Recovery_Controller, self).__init__()

        activation = torch.nn.ELU()

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.std = nn.Parameter(0 * torch.ones(num_actions))

        # self.distribution = None
        # # disable args validation for speedup
        # Normal.set_default_validate_args = False
        
    def forward(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def reset(self):
        None

class YangTraj(torch.nn.Module):
    def __init__(self, stand_time=2.0):
        super().__init__()
        
        self.time_step = 0.02
        self.stand_time = stand_time
        #                                         FL                FR              RL              RR
        self.default_dof_pos = torch.tensor([0.1, 0.8, -1.5, 
                                             -0.1, 0.8, -1.5, 
                                             0.1, 1.0, -1.5, 
                                             -0.1, 1.0, -1.5], device="cuda:0", dtype= torch.float32)
        self.yang_pos = torch.tensor([0.1, 0.8, -1.0, 
                                             -0.1, 0.8, -1.0, 
                                             0.1, 1.3, -1.87, 
                                             -0.1, 1.3, -1.87], device="cuda:0", dtype= torch.float32)
        self.init_motor_angles = torch.zeros_like(self.default_dof_pos, device="cuda:0")
        
        self.t = 0.0
        self.first = True

    def forward(self, relativa_dof_pos):
        if self.first:
            self.init_motor_angles = self.default_dof_pos + relativa_dof_pos
            self.first = False

        blend_ratio = min(self.t / self.stand_time, 1)
        self.t += self.time_step 
        
        action = (1 - blend_ratio)  * self.init_motor_angles + blend_ratio * self.yang_pos 
        
        return action - self.default_dof_pos
    
    def reset(self):
        self.t = 0.0
        self.first = True

class FuTraj(torch.nn.Module):
    def __init__(self, stand_time=2.0):
        super().__init__()
        
        self.time_step = 0.02
        self.stand_time = stand_time
        #                                         FL                FR              RL              RR
        self.default_dof_pos = torch.tensor([0.1, 0.8, -1.5, 
                                             -0.1, 0.8, -1.5, 
                                             0.1, 1.0, -1.5, 
                                             -0.1, 1.0, -1.5], device="cuda:0", dtype= torch.float32)
        self.fu_pos = torch.tensor([0.1, 0.6, -1.8, 
                                             -0.1, 0.6, -1.8, 
                                             0.1, 0.2, -1.0, 
                                             -0.1, 0.2, -1.0], device="cuda:0", dtype= torch.float32)
        self.init_motor_angles = torch.zeros_like(self.default_dof_pos, device="cuda:0")
        
        self.t = 0.0
        self.first = True

    def forward(self, relativa_dof_pos):
        if self.first:
            self.init_motor_angles = self.default_dof_pos + relativa_dof_pos
            self.first = False

        blend_ratio = min(self.t / self.stand_time, 1)
        self.t += self.time_step 

        action = (1 - blend_ratio)  * self.init_motor_angles + blend_ratio * self.fu_pos 
        
        return action - self.default_dof_pos
    
    def reset(self):
        self.t = 0.0
        self.first = True

