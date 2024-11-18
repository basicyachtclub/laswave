from setuptools import setup

setup()

# for compatibility with legacy builds or versions of tools that don’t support certain packaging standards

# pip may allow editable install only with pyproject.toml and setup.cfg. 
# However, this behavior may not be consistent over various pip versions and 
# other packaging-related tools (setup.py is more reliable on those scenarios).