from setuptools import setup, find_packages

setup(
    name='thsd',
    version='0.1',
    description='tweet-hate-speech-detection',
    author='Yamen',
    author_email='yamenahmadhabib@gmail.com',
    packages=find_packages(),
    install_requires=(
        'networkx>=2.3',
        'simpy>=3.0.11',
        'numpy>=1.15.3',
        'pyyaml>=4.2b1',
        'torch>=1.0.1'
    )
)

