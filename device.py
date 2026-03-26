import torch

# my application will use either the gpu or the cpu depending on the specification of the computer hardware that it be run on
# cuda is for Nvidia GPUs which is supported by pytorch
# mps is for the new Apple ARM GPUs for Mac devices


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)

