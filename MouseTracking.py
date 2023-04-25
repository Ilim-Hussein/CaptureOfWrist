import pyautogui
import cv2

class MouseTrack:

    def MouseControl (cordX, cordY):

          try:
              #pyautogui.FAILSAFE = False
              pyautogui.moveTo(cordX, cordY, 0.1)
              if cv2.waitKey(1) & 0xFF == ord('r'):
                pyautogui.leftClick()

          except pyautogui.FailSafeException:
              pyautogui.FAILSAFE = False


