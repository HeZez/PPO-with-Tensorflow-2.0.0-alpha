import core
from pprint import pprint
from mlagents.envs import UnityEnvironment
from utils.logger import log
from core.Env import UnityEnv, GymCartPole


class Manager:
    def __init__(self,
                 env_name="",
                 trainer=None,
                 train_params=None,
                 policy_params=None,
                 sil_params=None):

        self.env_name = env_name            # ENV to load
        self.trainer = trainer              # Trainer Policy from yaml config
        self.train_params = train_params    # Trainer Parameters from yaml config passed to trainer class on init
        self.policy_params = policy_params  # Policy Parameters from yaml config 
        self.sil_params = sil_params

        # Start ML Agents Environment | Without filename in editor training is started
        # self.env = GymCartPole() 
        self.env = UnityEnv(env_name=env_name,seed =train_params['seed'])
        

    def start(self):

        # Logging all about Trainer
        log(self.trainer + " loaded")
        log("Trainer Parameters")
        pprint(self.train_params, width=10, indent=5)

        # Get the trainer class for initialization
        trainer_class = getattr(core, self.trainer)
        trainer = trainer_class(**self.train_params, env=self.env, policy_params=self.policy_params, sil_params=self.sil_params)
        # Start the trainer
        trainer.start()
        # Close the Environment at the end
        self.env._get_env.close()
