from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="sisyphus-env",
    version="1.0.0",
    description="Gym environments used in our IROS paper \"Active Exploration for Robotic Manipulation\".",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tim Schneider",
    author_email="schneider@ias.tu-darmstadt.de",
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["*.urdf"]},
    python_requires=">=3.8",
    install_requires=[
        "gym>=0.17.2",
        "numpy>=1.22.0",
        "robot-gym @ git+https://github.com/TimSchneider42/robot-gym.git"
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
    ],
)
