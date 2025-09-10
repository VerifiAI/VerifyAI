#!/usr/bin/env python3

# This script will patch the app.py to add tracing to requests.get calls

import os

# Read the current app.py
with open('app.py', 'r') as f:
    content = f.read()

# Add tracing code after the imports
tracing_code = '''
# === DEBUG TRACING CODE ===
import traceback as tb_module
original_requests_get = requests.get

def traced_requests_get(*args, **kwargs):
    logger.error(f"TRACE: requests.get called with args: {args}")
    logger.error(f"TRACE: requests.get called with kwargs: {kwargs}")
    logger.error("TRACE: Call stack:")
    for line in tb_module.format_stack():
        logger.error(f"TRACE: {line.strip()}")
    return original_requests_get(*args, **kwargs)

requests.get = traced_requests_get
# === END DEBUG TRACING CODE ===

'''

# Find where to insert the tracing code (after the requests import)
lines = content.split('\n')
new_lines = []
inserted = False

for line in lines:
    new_lines.append(line)
    if 'import requests' in line and not inserted:
        new_lines.extend(tracing_code.split('\n'))
        inserted = True

# Write the modified content
with open('app_traced.py', 'w') as f:
    f.write('\n'.join(new_lines))

print("Created app_traced.py with request tracing enabled.")
print("You can now run: python3 app_traced.py to see where requests.get is called.")