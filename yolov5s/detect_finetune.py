from yolov5s import YOLOv5s
from weight import mapped_load 
from uti import make_yolo_grid
from general import non_max_suppression
import cv2
import torch

#########################ancors##########################################
anchors = [[10,13, 16,30, 33,23],      # P3/8                          ##
           [30,61, 62,45, 59,119],      # P4/16                        ##
           [116,90, 156,198, 373,326]]  # P5/32                        ##
                                                                       ##
anchor_grid = torch.tensor(anchors).float().view(3, 1, -1, 1, 1, 2)    ##
#print(anchor_grid.shape)  # torch.Size([3, 1, 3, 1, 1, 2])            ##
#########################################################################



model_yolov5s = torch.load("yolov5s_custom_fintune.pt")

### dataloader
path='/home/lijiaxin/SimModel/yolov5s/data_set/output/img_0_12.jpg'
input_img = cv2.imread(path)
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
input_img = torch.from_numpy(input_img).unsqueeze(0)
#b h w c  -> b c h w
input_img = input_img.permute(0, 3, 1, 2)
# 归一化
input_img = input_img.float() / 255.0




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_yolov5s.to(device)
input_img = input_img.to(device)

# for batch_normal and dropout
model_yolov5s.eval()

result = None
### 推理
with torch.no_grad():
    outputs = model_yolov5s(input_img)
    x = []
    grid = [] 
    stride = [8, 16, 32]
    for item in outputs:
        #print(item.shape)
        _, _ , ny, nx = item.shape
        item = item.view(1, 3, 85, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        x.append(item)
        grid.append(make_yolo_grid(nx, ny))

    y= []
    for i in range(len(x)):
        cur_x = x[i]
        cur_grid = grid[i]
        cur_stride = stride[i]
        cur_anchor_grid = anchor_grid[i]  # 1,3,1,1,2

        assert cur_grid.shape[-2] * cur_stride == 640 

        ######## same with loss function ###########
        cur_y = cur_x.sigmoid()
        cur_y[..., 0:2] = (cur_y[..., 0:2] * 2. - 0.5 + cur_grid.to(x[i].device)) * cur_stride  # xy
        cur_y[..., 2:4] = (cur_y[..., 2:4] * 2) ** 2 * cur_anchor_grid.to(x[i].device)  # wh

        y.append(cur_y.view(1, -1, 85))

    res = torch.cat(y, dim=1)


    result = non_max_suppression(res)

    print(result)

############ 可视化 ############

import numpy as np

# COCO 类别名（只取需要的，或者全部）
coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush']


# 画框
img_vis = cv2.imread(path)  # BGR 格式
for det in result:

    d =  det.cpu().numpy()  # N,6

    for i in range(d.shape[0]):
        x1, y1, x2, y2, conf, cls = d[i]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)
        
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        label = f"{coco_names[cls]} {conf:.2f}"
        cv2.putText(img_vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 保存
cv2.imwrite('result.jpg', img_vis)
print("已保存 result.jpg")

    
    





