from yolov5s import YOLOv5s
from data_set import create_dataloader
from loss_component import compute_loss
from weight import mapped_load
import torch

model_yolov5s = YOLOv5s()

### dataloader
path='/home/lijiaxin/coco128/images/train2017'
dataloader, dataset = create_dataloader(path, batch_size=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 加载预训练权重
pure_dict = torch.load('/home/lijiaxin/yolov5/yolov5s_state_dict.pt', 
                        map_location='cpu', weights_only=True)

mapped_load(model_yolov5s, pure_dict)

# #### 冻结backbone 只训练head
# # 冻结 backbone 和 neck，只训练检测头
# for name, param in model_yolov5s.named_parameters():
#     if 'detect_layer' not in name:
#         param.requires_grad = False

# # 确认效果
# trainable = [n for n, p in model_yolov5s.named_parameters() if p.requires_grad]
# print(f"可训练参数 ({len(trainable)}): {trainable}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_yolov5s.to(device)

# for batch_normal and dropout
model_yolov5s.train()


num_epochs = 500
### 进行训练
optimizer = torch.optim.AdamW(model_yolov5s.parameters(), lr=1e-4, weight_decay=5e-4)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)  先不使用scheduler

for epoch in range(num_epochs):
    model_yolov5s.train()
    for i, (img, label, path, shapes) in enumerate(dataloader):
        optimizer.zero_grad()
        img = img.to(device)
        label = label.to(device)

        img = img.float() / 255.0
        pred = model_yolov5s(img)
        pred = list(pred)
        for i in range(len(pred)):
            b , c , h , w = pred[i].shape
            #print(pred[i].shape)

            pred[i] = pred[i].view(b, 3, -1, h, w).permute(0, 1, 3, 4, 2)


        loss = compute_loss(pred, label)
        print(f"loss: {loss[0].item()}")
        if loss[0].item() < 0.01:
            break
        else:
            loss[0].backward()
            optimizer.step()
            #scheduler.step()

### 保存模型
torch.save(model_yolov5s, 'yolov5s_custom_fintune.pt')
