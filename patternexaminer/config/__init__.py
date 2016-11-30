import os
import configparser
config = configparser.ConfigParser()
conf_base = 'patternexaminer/config/'

def read_conf(filename):
    path = os.path.join(conf_base, filename)
    if os.path.isfile(path):
        config.read_file(open(path))

read_conf('base.ini')
read_conf('base_private.ini')

# When developing this app and wish to enjoy different 
# configuration files, set the environment vaiable.

if os.environ.get('DEPLOYMENT') == 'development':
    read_conf('dev.ini')
else:
    read_conf('prod.ini')
