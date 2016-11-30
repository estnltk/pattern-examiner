#!/usr/bin/env python3

from patternexaminer.database import init_db, populate_options, populatePublicData

from patternexaminer import database
init_db()
populate_options()
populatePublicData()

