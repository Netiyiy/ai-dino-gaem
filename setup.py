from setuptools import setup, find_packages

setup(
    name='ai_dino_game',  # Name of your project/package
    version='0.1',  # Version of your project/package
    packages=find_packages(),  # Automatically finds and includes packages
    install_requires=[
        'gymnasium',  # Dependencies
        'stable-baselines3',
    ],
)
