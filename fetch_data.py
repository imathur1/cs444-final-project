import os

files = [
    "2bdb0560cb6a424e868b32abeb0ff1d0-1626210507600004009.png",
    "4a509f79565747b7818df23b79cf5b7f-1657143534200002003.png",
    "c0bf31f67b8a41819a2044d3d9505333-1621531062300005397.png"
    ]

epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25]

models = ["dpt_swin2_large", "dpt_swin2_tiny", "midas_v21_small"]
types = ["", "_targeted"]

for model in models:
    for typ in types:
        for eps in epsilons:
            for file in files:
                command = f'gcloud compute scp ishaanmathur_16@cs444vm-vm:~/444/MiDaS/{model}{typ}/eps_{eps}/{file} viz/{model}{typ}/eps_{eps}'
                os.system(command)