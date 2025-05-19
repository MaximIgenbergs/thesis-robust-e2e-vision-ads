# global_manager.py
from multiprocessing import Manager
from .action import UdacityAction

# Lazy initialization of Manager and simulator_state
_manager = None
_simulator_state = None

def get_manager():
    global _manager
    if _manager is None:
        _manager = Manager()
    return _manager

def get_simulator_state():
    global _simulator_state
    if _simulator_state is None:
        manager = get_manager()
        _simulator_state = manager.dict({
            'observation': None,
            'action': UdacityAction(0.0, 0.0),
            'paused': False,
            'track': "lake",
            'events': [],
            'episode_metrics': None
        })
    return _simulator_state
