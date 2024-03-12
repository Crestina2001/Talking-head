import torch

checkpoint = torch.load("face_detection/detection/sfd/s3fd.pth", map_location="cpu")
print(checkpoint.keys())
