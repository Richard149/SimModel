import onnx

model = onnx.load('yolov5s_int8.onnx')

# 看所有节点类型
op_types = set(n.op_type for n in model.graph.node)
print("ONNX 中的算子类型:")
for op in sorted(op_types):
    count = sum(1 for n in model.graph.node if n.op_type == op)
    print(f"  {op}: {count}")

# 搜索量化相关
q_nodes = [n for n in model.graph.node if 'Quant' in n.op_type or 'quant' in n.op_type.lower()]
d_nodes = [n for n in model.graph.node if 'Dequant' in n.op_type or 'dequant' in n.op_type.lower()]
print(f"\nQuantize 节点: {len(q_nodes)}")
print(f"Dequant 节点: {len(d_nodes)}")

# 搜索 FakeQuant 相关
fq_nodes = [n for n in model.graph.node if 'fake' in n.name.lower() or 'Fake' in n.name]
print(f"FakeQuant 相关: {len(fq_nodes)}")