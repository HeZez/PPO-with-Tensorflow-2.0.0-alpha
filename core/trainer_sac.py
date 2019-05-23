import tensorflow as tf
import numpy as np
from pprint import pprint
import time
from core.SAC.buffer_sac import ReplayBuffer
from core.SAC.policy_sac import Policy_SAC
from core.Env import UnityEnv, Continuous, Discrete
from utils.logger import log, Logger 


class Trainer_SAC:

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

        self.start_steps = int(5000)
        self.batch_size = 128

        log("Policy Parameters")
        pprint(policy_params, indent=5, width=10)

        self.buffer_sac = ReplayBuffer(self.env.EnvInfo.obs_shape,self.env.EnvInfo.act_size, int(1e6))
        self.agent = Policy_SAC(policy_params= policy_params, env_info= self.env.EnvInfo)

        self.logger = Logger(logger_name)

        
    def start(self):

        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        if self.load_model:
            self.agent.load()
            self.agent.v_target.set_weights(self.agent.v.get_weights())

        if self.training:
            log("Starting Trainer ...", color="warning")
            self.train()
        else:
            log("Starting Inference ...", color="warning")
            self.inference()

    def test_agent(self, n=10):
        for _ in range(n):
            o, r, d = self.env.reset()
            ep_ret, ep_len  = 0, 0
            while not(d or (ep_len == self.max_episode_length)):
                # Take deterministic actions at test time 
                a = self.agent.pi.get_action(o, True)
                o, r, d = self.env.step(a)
                ep_ret += r
                ep_len += 1
            self.logger.store("TestEpRet", ep_ret)
            self.logger.store("TestEpLen", ep_len)

    # Main training loop
    def train(self):

        start_time = time.time()
        o, r, d = self.env.reset() 
        ep_ret, ep_len = 0, 0
        # total_steps = self.steps_per_epoch * self.epochs
        steps = 0

        # Main loop: collect expieriences in env
        for epoch in range(self.epochs):

            for step in range(self.steps_per_epoch):

                steps +=1

                if steps > self.start_steps:
                    a = self.agent.pi.get_action(o)
                else:
                    a = np.random.randn(self.env.EnvInfo.act_size) # random env action_space.sample()
                
                # Step the env
                o2, r, d= self.env.step(a)
                ep_ret += r
                ep_len += 1

                d = False if ep_len==self.max_episode_length else d

                self.buffer_sac.store(o, a, r, o2, d)

                # Super critical dont forget
                o = o2

                if d or(ep_len == self.max_episode_length) or (step == self.steps_per_epoch-1):

                    for _ in range(7):

                        batch = self.buffer_sac.sample_batch(self.batch_size)
                        pi_loss, q1_loss, q2_loss, v_loss = self.agent.update(batch, ep_len)
                            
                    self.logger.store("EpRet", ep_ret)
                    self.logger.store("EpLen", ep_len)

                    o, r, d = self.env.reset() 
                    ep_ret, ep_len = 0, 0

            # End of epoch wrap-up

            # Saving every n steps
            if (epoch % self.save_freq == 0) or (epoch == self.epochs - 1):
                self.agent.save()
               
                # Test the performance of the deterministic version of the agent.
                print("Testing....")
                self.test_agent()
               

            # Log info about epoch
            self.logger.store('Epoch', epoch)
            self.logger.store('Total Steps', steps)
            self.logger.store('LossPi', pi_loss)
            self.logger.store('LossQ1', q1_loss)
            self.logger.store('LossQ2', q2_loss)
            self.logger.store('LossV', v_loss)
            self.logger.store('Time', time.time()-start_time)
            self.logger.log_metrics(epoch)

            # self.logger.store('Q1Vals') 
            # self.logger.store('Q2Vals')         
            # self.logger.store('VVals') 
            # self.logger.store('LogPi')

    def inference(self):

        o, r, d = self.env.reset()
        ep_ret, ep_len = 0, 0
         
        for epoch in range(self.epochs):

            for step in range(self.steps_per_epoch):

                a = self.agent.pi.get_action(o, True)
                          
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

    