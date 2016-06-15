# -*- coding: utf-8 -*-
import json

# load settings json
f = open('settings.json')
data = json.load(f)

# parse json to python variables
for key, value in data.items():
    globals()[key] = value
