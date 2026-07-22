from yolov5s_qat import YOLOv5s_Q
from data_set import create_dataloader
from loss_component import compute_loss
from weight import mapped_load
import torch
from torch.ao.quantization import FakeQuantize

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/qat")

model_yolov5s = YOLOv5s_Q()

### dataloader
path='/home/lijiaxin/coco128/images/train2017'
dataloader, dataset = create_dataloader(path, batch_size=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#加载finetune weight
weight_path = "yolov5s_custom_fintune.pt"
check_point = torch.load(weight_path ,  map_location='cpu')

model_yolov5s.load_state_dict(check_point.state_dict())

#### fuse
model_yolov5s.eval()
model_yolov5s = model_yolov5s.fuse()


# for batch_normal and dropout
model_yolov5s.train()

####### 插入量化节点
# 改成
from torch.quantization import QConfig, MinMaxObserver, MovingAverageMinMaxObserver

model_yolov5s.qconfig = QConfig(
    activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_affine),
    weight=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_affine , dtype=torch.qint8)
)
model_yolov5s = torch.quantization.prepare_qat(model_yolov5s)


##############normal-train############################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_yolov5s.to(device)




num_epochs = 5
### 进行训练
optimizer = torch.optim.AdamW(model_yolov5s.parameters(), lr=1e-4, weight_decay=5e-4)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)  先不使用scheduler

#### 监控前几层的QAT的scale的变化情况
qat_layers = ['focus_0.conv.conv', 'conv_1', 'c3_2.cv1']

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

        ############ 监控前几层的scale
        if i % 1 == 0:
            step = epoch * len(dataloader) + i
            for name, module in model_yolov5s.named_modules():
                for ql in qat_layers:
                    if ql in name and name.endswith('weight_fake_quant') and isinstance(module, FakeQuantize):
                        s = module.scale  # FakeQuantize.scale 存在
                        writer.add_scalar(f'scale/{ql}_mean', s.mean().item(), step)
                        writer.add_scalar(f'scale/{ql}_min',  s.min().item(), step)
                        writer.add_scalar(f'scale/{ql}_max',  s.max().item(), step)


import os

# ====== 1. 保存 QAT 权重 ======
model_yolov5s.eval()
torch.save(model_yolov5s.state_dict(), 'checkpoint/qat_state.pt')
print(f"QAT 权重已保存: checkpoint/qat_state.pt")

####### 导出量化友好的ONNX模型
from yolov5s import YOLOv5s

model = YOLOv5s()  ###不带量化节点的
model.to("cpu")

model.eval()
model.fuse()

# 加载 QAT 权重，只取匹配的 key（过滤掉量化相关的 key）
qat_state = torch.load('checkpoint/qat_state.pt', map_location='cpu')

# 过滤：只保留模型中存在的 key（去掉 quant/dequant/fake_quant 相关）
model_keys = set(model.state_dict().keys())
filtered_state = {k: v for k, v in qat_state.items() if k in model_keys}

# 加载过滤后的权重
missing, unexpected = model.load_state_dict(filtered_state, strict=False)
if missing:
    print(f"缺失的 key（已忽略，是量化相关参数）: {len(missing)}")
if unexpected:
    print(f"多余的 key: {unexpected}")


model = model.cpu()

# 用固定的 dummy input
dummy_input = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    model,
    dummy_input,
    'yolov5s_qat.onnx',
    opset_version=13,
    input_names=['images'],
    output_names=['output_0', 'output_1', 'output_2'],
)

print(f"ONNX 已保存: yolov5s_qat.onnx ({os.path.getsize('yolov5s_qat.onnx')/1e6:.2f} MB)")



