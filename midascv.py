import cv2
import torch
from matplotlib import pyplot as plt

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to('cpu')
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    cv2.imshow('CV2 video', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()