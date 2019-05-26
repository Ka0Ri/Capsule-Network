import sys
import os
import numpy as np
import torch

from torch.autograd import Variable
import cv2
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from capsule_network import *


__app_name__ = 'capsule_test'

form_class = uic.loadUiType("main_window.ui")[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle(__app_name__)
        self.img_boxes = [self.label_100, self.label_101, self.label_102, self.label_103, self.label_104,
        self.label_105, self.label_106, self.label_107, self.label_108, self.label_109,
        self.label_110, self.label_111, self.label_112, self.label_113, self.label_114,
        self.label_115, self.label_116, self.label_117, self.label_118, self.label_119,
        self.label_120, self.label_121, self.label_122, self.label_123, self.label_124]

        self.change = np.zeros((25,10,16))
        self.sliders = [self.horizontalSlider_1, self.horizontalSlider_2, self.horizontalSlider_3, self.horizontalSlider_4,
        self.horizontalSlider_5, self.horizontalSlider_6, self.horizontalSlider_7, self.horizontalSlider_8,
        self.horizontalSlider_9, self.horizontalSlider_10, self.horizontalSlider_11, self.horizontalSlider_12,
        self.horizontalSlider_13, self.horizontalSlider_14, self.horizontalSlider_15, self.horizontalSlider_16] 
        
        self.pushButton_reset.clicked.connect(self.reset)
        for i in range(25):
            self.img_boxes[i].setScaledContents(True)
       
        self.model = CapsuleNet()
        self.model.load_state_dict(torch.load('epochs/epoch_87.pt', map_location='cpu'))
        test_sample = next(iter(get_iterator(False)))
        self.ground_truth = (test_sample[0].unsqueeze(1).float() / 255.0)
        
        self.x, self.classes = self.model(Variable(self.ground_truth))
        reconstruction = self.reconstruct()
        
        for i in range(25):
            np_img = 255*np.array(reconstruction[i][0])
            np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
            cv2.imwrite("gray.jpg", np_img)
            self.img_boxes[i].setPixmap(QPixmap('gray.jpg'))
        
        for i in range(16):
            self.sliders[i].valueChanged.connect(self.valuechange)
        
    def valuechange(self):
        for i in range(16):
            self.change[:,:,i] = 0.25*self.sliders[i].value()/100

        reconstruction = self.reconstruct()
        for i in range(25):
            np_img = 255*np.array(reconstruction[i][0])
            np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
            cv2.imwrite("gray.jpg", np_img)
            self.img_boxes[i].setPixmap(QPixmap('gray.jpg'))

    def reconstruct(self):
        y = torch.tensor(self.change).float()
        #rotation matrix
        # R_t = np.identity(16)
        # for i in range(15):
        #     R = np.identity(16)
        #     R[i,i] = np.cos(self.change[i])
        #     R[i + 1, i + 1] = np.cos(self.change[i])
        #     R[i, i + 1] = -np.sin(self.change[i])
        #     R[i + 1, i] = np.sin(self.change[i])
        #     R_t = R @ R_t
        # R = torch.from_numpy(R_t.T).float()
        z = self.x + y
        
        # squared_norm = (z ** 2).sum(dim=-1, keepdim=True)
        # scale = squared_norm / (1 + squared_norm)
        # v = scale * z / torch.sqrt(squared_norm)
        v = z

        _, max_length_indices = self.classes.max(dim=1)
        cl = Variable(torch.eye(NUM_CLASSES)).index_select(dim=0, index=max_length_indices)
        v = (v * cl[:, :, None])
        

        reconstructions = self.model.decoder(v.view(self.x.size(0), -1))
        reconstruction = reconstructions.view_as(self.ground_truth).data
        return reconstruction

    def reset(self):
        for i in range(16):
            self.sliders[i].setValue(0)
        self.valuechange()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Cleanlooks"))
    app.setPalette(QApplication.style().standardPalette())
    myWindow = MyWindow()
    myWindow.show()
    sys.exit(app.exec_())
