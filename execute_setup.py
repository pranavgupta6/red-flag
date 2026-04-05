#!/usr/bin/env python
import os
import sys

# Execute the setup_project.py script
exec_globals = {}
with open(r'D:\zz projects\red-flag\setup_project.py', 'r') as f:
    script_content = f.read()
    exec(script_content, exec_globals)

print("Setup script execution completed.")
