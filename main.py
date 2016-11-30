#!/usr/bin/env python3

import configparser
from patternexaminer import app
from patternexaminer.config import config

app.run(host=config.get('WEB_SERVER','HOST'), port=config.getint('WEB_SERVER','PORT'), threaded=True, debug=True)
