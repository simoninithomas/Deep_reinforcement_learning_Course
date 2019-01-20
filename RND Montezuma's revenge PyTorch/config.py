import configparser

config = configparser.ConfigParser()
config.read('./config.conf')

# ---------------------------------
default = 'DEFAULT'
# ---------------------------------
default_config = config[default]
