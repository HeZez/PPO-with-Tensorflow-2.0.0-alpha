import tensorflow as tf
import numpy as np
from pprint import pprint
import time
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

        self.start_steps = start_steps

        log("Policy Parameters")
        pprint(policy_params, indent=5, width=10)

        # self.buffer_ppo = Buffer_PPO(self.steps_per_epoch, self.env.EnvInfo, gamma= self.gamma, lam= self.lam)
        # self.agent = Policy_PPO(policy_params= policy_params, env_info= self.env.EnvInfo)

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
            # self.inference()


    # Main training loop
    def train(self):

        start_time = time.time()
        o, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0
        total_steps = self.steps_per_epoch * self.epochs

        # Main loop: collect expieriences in env
        for t in range(total_steps):

            if t > self.start_steps:
                a = self.get_action(o)
            else:
                a = self.env.action_space.sample()
            
            # Step the env
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            d = False if ep_len==self.max_ep_len else d

            self.replay_buffer.store(o, a, r, o2, d)

            # Super critical dont forget
            o = o2

            if d or(ep_len == self.max_ep_len):

                for j in range(ep_len):

                    batch = self.replay_buffer.sample_batch(self.batch_size)

                    feed_dict = {self.x_ph: batch['obs1'],
                                self.x2_ph: batch['obs2'],
                                self.a_ph: batch['acts'],
                                self.r_ph: batch['rews'],
                                self.d_ph: batch['done'],
                                }
                    outs = self.sess.run(self.step_ops, feed_dict)

                    self.logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
                                    VVals=outs[6], LogPi=outs[7])
                        
                    
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0

            # End of epoch wrap-up
            if t > 0 and t % self.steps_per_epoch == 0:
                
                epoch = t // self.steps_per_epoch

                self.logger.store(TestEpRet=0, TestEpLen=0)

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs-1):
                    self.logger.save_Model(self.sess, step=t)

                # Test the performance of the deterministic version of the agent.
                if self.logger.get_stats('EpRet') > 0.1:
                    print("Testing....")
                    self.test_agent()
                    #self.bRenderFlag = True

                # Log info about epoch
                self.logger.store('Epoch', epoch)
                self.logger.store('EpRet')
                self.logger.store('TestEpRet') 
                self.logger.store('EpLen')
                self.logger.store('TestEpLen')
                self.logger.store('TotalEnvInteracts', t)
                self.logger.store('Q1Vals') 
                self.logger.store('Q2Vals')         
                self.logger.store('VVals') 
                self.logger.store('LogPi')
                self.logger.store('LossPi')
                self.logger.store('LossQ1')
                self.logger.store('LossQ2')
                self.logger.store('LossV')
                self.logger.store('Time', time.time()-start_time)
                self.logger.log_metrics()

            
    # def inference(self):

    #     o, r, d = self.env.reset()
    #     ep_ret, ep_len = 0, 0
         
    #     for epoch in range(self.epochs):

    #         for step in range(self.steps_per_epoch):

    #             a, _ = self.agent.pi.get_action_logp(o)
                          
    #             # make step in env
    #             o, r, d = self.env.step(a)
                  
    #             ep_ret += r
    #             ep_len += 1

    #             terminal =  d or (ep_len == self.max_episode_length)

    #             if terminal or (step == self.steps_per_epoch-1):

    #                 if terminal and ep_len > 10:
    #                     self.logger.store('Rewards', ep_ret)
    #                     self.logger.store('Eps Length', ep_len)

    #                 o, r, d = self.env.reset()
    #                 ep_ret, ep_len = 0, 0
      
    #         self.logger.log_metrics(epoch)

    