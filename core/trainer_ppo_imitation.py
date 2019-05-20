import tensorflow as tf
import numpy as np
from pprint import pprint
from core.PPO.buffer_ppo import Buffer_PPO
from core.Imitation.buffer_behavioral_cloning import Buffer_Imitation
from core.PPO.policy_ppo import Policy_PPO
from core.Imitation.policy_behavioral_cloning import Behavioral_Cloning
from core.Env import UnityEnv, Continuous, Discrete
from utils.logger import log, Logger 


class Trainer_PPO_IMITATION:

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
                 bc_params=dict(),
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
        self.bc_params = bc_params

        self.batch_size_bc = self.bc_params['batch_size_bc']

        log("Policy Parameters")
        pprint(policy_params, indent=5, width=10)

        self.buffer_ppo = Buffer_PPO(self.steps_per_epoch, self.env.EnvInfo, gamma= self.gamma, lam= self.lam)
        self.agent = Policy_PPO(policy_params= policy_params, env_info= self.env.EnvInfo)

        self.buffer_imitation = Buffer_Imitation(steps_per_epoch, self.env.EnvInfo)
        self.imitation = Behavioral_Cloning(**self.bc_params, pi = self.agent.pi, optimizer_pi = self.agent.optimizer_pi, num_actions= self.env.num_actions)

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

        
        o, r, d, o_teacher, prev_act_teacher = self.env.reset() 
        ep_ret, ep_len = 0, 0
         
        for epoch in range(self.epochs):

            for step in range(self.steps_per_epoch):

                a, logp_t = self.agent.pi.get_action_logp(o)
                v_t = self.agent.v.get_value(o)           
                 
                self.buffer_ppo.store(o, a, r, v_t, logp_t)
                self.buffer_imitation.store(o_teacher, prev_act_teacher)
                    
                # make step in env
                o, r, d, o_teacher, prev_act_teacher = self.env.step(a)
                
                ep_ret += r
                ep_len += 1

                terminal =  d or (ep_len == self.max_episode_length)

                if terminal or (step == self.steps_per_epoch-1):

                    if not terminal:
                        log('Warning: trajectory was cut off by epoch at %d steps.' %(ep_len))
                        
                    last_val = r if d else self.agent.v.get_value(o)
                    self.buffer_ppo.finish_path(last_val)

                    if terminal: 
                        self.logger.store('Rewards', ep_ret)
                        self.logger.store('Eps Length', ep_len)

                    o, r, d, o_teacher, prev_act_teacher = self.env.reset() 
                    ep_ret, ep_len = 0, 0

                # END OF FOR STEP LOOP

            # Update via PPO
            obs, act, adv, ret, logp_old = self.buffer_ppo.get()
            loss_pi, loss_entropy, approx_ent, kl, loss_v = self.agent.update(obs,act,adv, ret, logp_old)

            # Logging PPO
            self.logger.store('Pi Loss', loss_pi)
            self.logger.store('Ent Loss', loss_entropy)
            self.logger.store('Approx Ent', approx_ent)
            self.logger.store('KL', kl)
            self.logger.store('V Loss', loss_v)

            # Update via Imitation Learning
            im_obs, im_act = self.buffer_imitation.sample(self.batch_size_bc)
            if len(im_act) > 0:
                log('Updating via Behavioral Cloning ...')
                loss_bc = self.imitation.update_BC(im_obs, im_act)
                self.logger.store('BC Loss', loss_bc)
  
            # Saving every n steps
            if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
                self.agent.save()
            
            # Dump Logs
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

    