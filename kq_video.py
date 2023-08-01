import torch 
import cv2 
import numpy as np
from torchvision import transforms as T
from model_decoder_encoder import MobileNetV2_Segmentation
import torch.nn.functional as F
import time
color_map = [(128, 64,128),
             (255,223,0),
             ( 70, 70, 70),
             (102,102,156),
             (190,153,153),
             (153,153,153),
             (250,170, 30),
             (220,220,  0),
             (107,142, 35),
             (152,251,152),
             ( 70,130,180),
             (220, 20, 60),
             (255,  0,  0),
             (  0,  0,142),
             (  0,  0, 70),
             (  0, 60,100),
             (  0, 80,100),
             (  0,  0,230),
             (119, 11, 32)]
def preprocess(img:np.ndarray):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]   
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(1280,960))
    t = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
    img = t(img)
    return img
if __name__ =="__main__":
    # img=cv2.imread("images/2080.jpg")
    vid=cv2.VideoCapture("IMG_0302.MOV")
    output_file = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    frame_size = (1280,960)
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
    count=0
    while True: #  
        ret,img=vid.read()
        sv_img = np.zeros_like(img).astype(np.uint8)
        sv_img=cv2.resize(sv_img,(1280,960))
        img_re=preprocess(img)
        img_re=img_re
        print(img_re.shape)
        model=torch.load("./weights_TQB/best_model.pt")
        model.eval().cpu()
        img_re=img_re.unsqueeze(0)
        tik=time.perf_counter()
        output=model(img_re)
        pred = F.interpolate(output, size=img_re.size()[-2:], 
                                mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
        print("Time processing:",time.perf_counter()-tik)
        mask=np.where(pred==2,255,0).astype(np.uint8)
        color_map = cv2.applyColorMap(mask, cv2.COLORMAP_VIRIDIS)
        out.write(color_map)
    
    
    
    
    