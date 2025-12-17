python3.9 -m venv envs/.venv-udc
source envs/.venv-udc/bin/activate
cd external/perturbation-drive
pip install -r requirements_linux.txt
pip install "python-socketio==4.5.1" "python-engineio==3.13.2"
pip install -e .
pip install -e ../..
pip install "eventlet==0.35.1" "Flask-SocketIO==4.3.1" "Flask==2.0.0" "Werkzeug==2.0.3" "gymnasium==0.29.1"
pip install lightning