from setuptools import setup, find_packages

MAJOR = 0
MINOR = 0
MICRO = 1
VERSION = f'{MAJOR}.{MINOR}.{MICRO}'

STATUSES = [
    "1 - Planning",
    "2 - Pre-Alpha",
    "3 - Alpha",
    "4 - Beta",
    "5 - Production/Stable",
    "6 - Mature",
    "7 - Inactive"
]

DESCRIPTION = 'Udacity Gym Python Library'
LONG_DESCRIPTION = 'Python Library for Udacity Gym'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="udacity_gym",
    version=VERSION,
    author="Davide Yi Xian Hu",
    author_email="<davideyi.hu@polimi.it>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        'gymnasium>=0.29.1',
        "Flask==2.0.0",
        "Flask-SocketIO==4.3.1",
        "python-engineio==3.13.2",
        "python-socketio==4.5.1",
        "eventlet==0.35.1",
        "pandas>=1.5.3",
        "Werkzeug==2.0.3",
        "pillow==10.2.0",
        "tqdm>=4.66.4"
    ],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'

    keywords=['udacity', 'gym', 'simulator'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Driving Simulator",
    ]
)
