import core
from pprint import pprint
from mlagents.envs import UnityEnvironment
from utils.logger import log
from core.Env import UnityEnv, GymCartPole
import yaml, os, time


class Manager:
    def __init__(self,
                 env_name="",
                 trainer=None,
                 train_params=None,
                 policy_params=None,
                 config=None):

        self.env_name = env_name            # ENV to load
        self.trainer = trainer              # Trainer Policy from yaml config
        self.train_params = train_params    # Trainer Parameters from yaml config passed to trainer class on init
        self.policy_params = policy_params  # Policy Parameters from yaml config 

        # Start ML Agents Environment | Without filename in editor training is started
        # self.env = GymCartPole() 
        self.env = UnityEnv(env_name=env_name,seed =train_params['seed'])

        # make path for saving
        academy_name = self.env.get_env_academy_name
        named_tuple = time.localtime()
        time_string = time.strftime("%m_%d_%Y_%H_%M_%S", named_tuple)
        self.save_name = academy_name + "_" + time_string
        self.path = "./tmp/" + self.save_name + "/"
        
        os.makedirs(self.path, exist_ok=True)

        with open(self.path + academy_name + ".yaml", 'w+') as file:
            yaml.safe_dump(config, file)

    def start(self):

        # Logging all about Trainer
        log(self.trainer + " loaded")
        log("Trainer Parameters")
        pprint(self.train_params, width=10, indent=5)

        # Get the trainer class for initialization
        trainer_class = getattr(core, self.trainer)
        trainer = trainer_class(**self.train_params, env=self.env, policy_params=self.policy_params, logger_name= self.save_name)
        # Start the trainer
        trainer.start()
        # Close the Environment at the end
        self.env.env.close()
