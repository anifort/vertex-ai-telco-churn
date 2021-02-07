from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [''] # If execution failes due to missing packages, you can include them here as a list
 
setup(
    name='trainer', 
    version='0.1', 
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(), # Automatically find packages within this directory or below.
    include_package_data=True, # if packages include any data files, those will be packed together.
    description='Classification training for telco churn prediction model'
)