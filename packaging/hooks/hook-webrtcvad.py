from PyInstaller.utils.hooks import copy_metadata

# Pilot uses webrtcvad-wheels, which supplies the webrtcvad module.
datas = copy_metadata("webrtcvad-wheels")
