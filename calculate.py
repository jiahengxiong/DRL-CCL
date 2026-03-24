import json

with open("flow_model.json") as f:
    flows = json.load(f)["FlowModels"]

cnt0 = sum(1 for f in flows if len(f["child_flow_id"]) == 0)
cnt1 = sum(1 for f in flows if len(f["child_flow_id"]) == 1)

print("child_flow_id 长度 = 0:", cnt0)
print("child_flow_id 长度 = 1:", cnt1)
print("总 flow 数:", len(flows))