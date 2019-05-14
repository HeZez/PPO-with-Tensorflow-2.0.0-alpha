import tensorflow as tf
import numpy as np
from pprint import pprint

from core.PPO.buffer import Buffer_PPO, Buffer_Imitation
from core.PPO.policy_categorical import Policy_PPO_Categorical
from core.PPO.policy_continuous import Policy_PPO_Continuous
from core.SIL.policy_sil import SIL
from core.Imitation.Behavioral_Cloning import Behavioral_Cloning
from core.Env import UnityEnv
from utils.logger import log, Logger 


class Trainer_PPO:
    def __init__(self,
                 env=UnityEnv,
                 epochs=10,
                 steps_per_epoch=1000,
                 max_episode_length=1000,
                 gamma=0.99,
                 lam=0.97,
                 seed=0,
                 training=True,
                 load_model=False,
                 save_freq=1,
                 policy_params=dict(),
                 sil_params=dict(),
                 logger_name="",
                 **kwargs):

        self.env = env
        self.epochs= epochs
        self.steps_per_epoch = steps_per_epoch
        self.max_episode_length =max_episode_length
        self.gamma = gamma
        self.lam = lam
        self.seed= seed
        self.training = training
        self.load_model = load_model
        self.save_freq = save_freq
        self.policy_params = policy_params
        self.sil_params = sil_params

        log("Policy Parameters")
        pprint(policy_params, indent=5, width=10)

        self.buffer = Buffer_PPO(steps_per_epoch, obs_size= self.env.num_obs, act_size= self.env.num_actions, 
                                    act_type= self.env.action_space_type, gamma= gamma, lam= lam, is_visual = self.env.is_visual)

        self.imitation_buf = Buffer_Imitation(steps_per_epoch, obs_size=self.env.num_obs, act_size=self.env.num_actions,
                                                act_type = self.env.action_space_type, is_visual=self.env.is_visual)

        if self.env.action_space_type == 'discrete':

            self.agent = Policy_PPO_Categorical(policy_params= policy_params, num_actions= self.env.num_actions, is_visual = self.env.is_visual)

            if self.sil_params['use_sil']:

                self.SIL = SIL(**self.sil_params, pi =self.agent.pi, v= self.agent.v, 
                                optimizer_pi = self.agent.optimizer_pi, optimizer_v = self.agent.optimizer_v, num_actions = self.env.num_actions)

        elif self.env.action_space_type == 'continuous':

            self.agent = Policy_PPO_Continuous(policy_params= policy_params, num_actions= self.env.num_actions)

        if self.env.is_behavioral_cloning:
            self.imitation = Behavioral_Cloning(pi = self.agent.pi, optimizer_pi = self.agent.optimizer_pi, num_actions= self.env.num_actions)

        self.logger = Logger(logger_name)

        
    def start(self):

        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        if self.load_model:
            self.agent.load()

        if self.training:
            log("Starting Trainer ...", color="warning")
            self.train()
        else:
            log("Starting Inference ...", color="warning")
            self.inference()


    # Main training loop
    def train(self):

        if self.env.is_behavioral_cloning:
            o, r, d, o_teacher, prev_act_teacher = self.env.reset() 
            prev_obs_teacher = o_teacher
        else:
            o, r, d = self.env.reset()
        
        ep_ret, ep_len = 0, 0
         
        for epoch in range(self.epochs):

            for step in range(self.steps_per_epoch):

                a, logp_t = self.agent.pi.get_action_logp(o)
                v_t = self.agent.v.get_value(o)           
                 
                self.buffer.store(o, a, r, v_t, logp_t)
                
                # make step in env
                if self.env.is_behavioral_cloning:
                    o, r, d, o_teacher, prev_act_teacher = self.env.step(a)
                else:
                    o, r, d = self.env.step(a)
                  
                ep_ret += r
                ep_len += 1

                terminal =  d or (ep_len == self.max_episode_length)

                if terminal or (step == self.steps_per_epoch-1):

                    if not terminal:
                        log('Warning: trajectory was cut off by epoch at %d steps.' %(ep_len))
                        
                    last_val = r if d else self.agent.v.get_value(o)
                    self.buffer.finish_path(last_val)

                    if terminal: 
                        self.logger.store('Rewards', ep_ret)
                        self.logger.store('Eps Length', ep_len)

                        if self.sil_params['use_sil']:
                            trajectory = self.buffer.get_trajectory()
                            self.SIL.add_episode_to_per(trajectory)

                    if self.env.is_behavioral_cloning:
                        o, r, d, o_teacher, prev_act_teacher = self.env.reset() 
                    else:
                        o, r, d = self.env.reset()
                    
                    ep_ret, ep_len = 0, 0

                # Imitation Learning
                if self.env.is_behavioral_cloning:

                    self.imitation_buf.store(prev_obs_teacher, prev_act_teacher)
                    # remember previous observation
                    prev_obs_teacher = o_teacher

            # Update via PPO
            obs, act, adv, ret, logp_old = self.buffer.get()
            loss_pi, loss_entropy, approx_ent, kl, loss_v = self.agent.update(obs,act,adv, ret, logp_old)

            # Update via Imitation Learning
            if self.env.is_behavioral_cloning:

                im_batch_size = 200
                im_obs, im_act = self.imitation_buf.sample(im_batch_size)

                if len(im_act) > 0:
                    print("Updating via Imitation ....")
                    self.imitation.update_BC(im_obs, im_act)

            # Update via Self Imitation Learning
            if self.sil_params['use_sil']:
                
                self.SIL.update_SIL()
                
            # Saving every n steps
            if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
                self.agent.save()
            
            # Logging
            self.logger.store('Pi Loss', loss_pi)
            self.logger.store('Ent Loss', loss_entropy)
            self.logger.store('Approx Ent', approx_ent)
            self.logger.store('KL', kl)
            self.logger.store('V Loss', loss_v)
            self.logger.log_metrics(epoch)

            
    def inference(self):

        o, r, d = self.env.reset()
        ep_ret, ep_len = 0, 0
         
        for epoch in range(self.epochs):

            for step in range(self.steps_per_epoch):

                a, _ = self.agent.pi.get_action_logp(o)
                          
                # make step in env
                o, r, d = self.env.step(a)
                  
                ep_ret += r
                ep_len += 1

                terminal =  d or (ep_len == self.max_episode_length)

                if terminal or (step == self.steps_per_epoch-1):

                    if terminal and ep_len > 10:
                        self.logger.store('Rewards', ep_ret)
                        self.logger.store('Eps Length', ep_len)

                    o, r, d = self.env.reset()
                    ep_ret, ep_len = 0, 0
      
            self.logger.log_metrics(epoch)

    