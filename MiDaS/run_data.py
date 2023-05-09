import os

commands = [
    "python3 run.py --model_type midas_v21_small_256 --input_path inputs/ --output_path midas_v21_small_targeted/",
    "python3 run.py --model_type dpt_swin2_tiny_256 --input_path inputs/ --output_path dpt_swin2_tiny_targeted/",
    "python3 run.py --model_type dpt_swin2_large_384 --input_path inputs/ --output_path dpt_swin2_large_targeted/"
]

for command in commands:
    os.system(command)
    
os.chdir("/home/ishaanmathur_16/444/suadd23-monodepth-amazon/monosuadd/")

epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
models = ["midas_v21_small_targeted", "dpt_swin2_tiny_targeted", "dpt_swin2_large_targeted"]
for model in models:
    for eps in epsilons:
        command = f"python3 evaluate.py -gt ../../MiDaS/depth_annotations/ -pred ../../MiDaS/{model}/eps_{eps}/ -o ."
        os.system(command)