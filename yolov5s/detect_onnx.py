import onnxruntime as ort
import numpy as np
import cv2

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

# 1. 加载 ONNX 模型
session = ort.InferenceSession('yolov5s_qat.onnx')
print(f"输入名称: {session.get_inputs()[0].name}")
print(f"输出名称: {[o.name for o in session.get_outputs()]}")

# 2. 准备输入（和之前一样的图片）
path = '/home/lijiaxin/SimModel/yolov5s/data_set/output/img_0_13.jpg'
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640, 640))  # 确保是 640x640
img_input = img.astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2, 0, 1))  # HWC → CHW
img_input = np.expand_dims(img_input, axis=0)   # 加 batch 维度

print(f"输入 shape: {img_input.shape}")

# 3. 推理
outputs = session.run(None, {session.get_inputs()[0].name: img_input})

## 将outputs 转成 torch.Tensor
for i in range(len(outputs)):
    outputs[i] = torch.from_numpy(outputs[i])

result = None
x = []
grid = [] 
stride = [8, 16, 32]
for item in outputs:
    print(item.shape)
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
    d =  det.cpu().numpy()  # 1,6

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
cv2.imwrite('result_onnx.jpg', img_vis)
print("已保存 result_onnx.jpg")
