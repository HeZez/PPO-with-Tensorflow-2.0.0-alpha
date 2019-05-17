import os, yaml, click, logging
from manager import Manager

# Load yaml file from click argument and pass it to the manager instance
# Start manager instance
#
@click.command()
@click.argument('config')
def run(config=None):
    with open(config) as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    # print(config)
    m = Manager(**config, config=config)
    m.start()

# Main Entry Point
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()