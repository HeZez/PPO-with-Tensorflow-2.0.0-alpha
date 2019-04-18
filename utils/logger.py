import tensorflow as tf
import numpy as np
import time


color2code = dict(
    warning='\033[1;41m',
    ok='\033[1;32;40m',
    info='\033[1;36;40m'
)


def log(str="", color="info"):
    col = color2code[color]
    end = "\033[0m" + "\n"
    dash = '-'*60
    print("\n" + col + dash)
    print(str)
    print(dash + end)

def logStr(str=""):
    col = color2code['ok']
    end = "\033[0m"
    print(col + str + end)


class Logger():

    def __init__(self, academy_name=''):

        self.metrics = dict()

        file_name_str = academy_name + '__' + str(time.time())
        self.path = ''
        self.summary_writer = tf.summary.create_file_writer('./tmp/summaries/' + file_name_str)


    def store(self, name, value):

        if name not in self.metrics.keys():
            self.metrics[name] = tf.keras.metrics.Mean(name=name)
        self.metrics[name].update_state(value)


    def log_metrics(self, step):

        log('MEAN METRICS START', color="ok")

        logStr('{:<10s}{:>10}'.format("Epoch", step))

        if not self.metrics:
            logStr('NO METRICS')
            
        else:
            for key, metric in self.metrics.items():
                value = metric.result()
                logStr('{:<10s}{:>10.5f}'.format(key, value))

                with self.summary_writer.as_default():
                    tf.summary.scalar(key, value, step=step)
                
                metric.reset_states()
        
        log('MEAN METRICS END', color="ok")


       
                

        




