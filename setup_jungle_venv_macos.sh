cd ~/thesis-robust-e2e-vision-ads/external/perturbation-drive
pip install -r requirements_macos.txt # select the correct requirements file for your OS
pip install --only-binary=:all: "grpcio==1.56.2" "h5py==3.10.0" --no-cache # to avoid building from source
pip install "python-socketio==4.5.1" "python-engineio==3.13.2"
pip install -e .
pip install -e ../..
pip install "eventlet==0.35.1" "Flask-SocketIO==4.3.1" "Flask==2.0.0" "Werkzeug==2.0.3" "gymnasium==0.29.1"