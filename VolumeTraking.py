from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
#volume.GetVolumeRange()
#volume.SetMasterVolumeLevel(-36.0, None)


class VolumeTracking:

    def VolumeControl(distance,minVol,maxVol):

        vol = np.interp(distance, [5, 105], [minVol, maxVol])
        #print(vol)
        volume.SetMasterVolumeLevel(vol, None)

