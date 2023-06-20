import os
import pathlib
from io import open
from setuptools import setup, find_packages

from my_package import config

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

requirements = open("requirements.txt").read()
# GitHub Private Repository
if '{GH_TOKEN}' in requirements:
    GH_TOKEN = os.environ['GH_TOKEN']
    requirements = requirements.replace('{GH_TOKEN}', GH_TOKEN)
requirements = requirements.splitlines()
print(requirements)

setup(
    name=config.PACKAGE_NAME,
    version=config.PACKAGE_VERSION,
    url=config.__URL__,
    author=config.__DS__,
    author_email = config.__DS_EMAIL__,
    maintainer = config.__MLE__,
    maintainer_email = config.__MLE_EMAIL__,
    description=config.__DESCRIPTION__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=requirements
)