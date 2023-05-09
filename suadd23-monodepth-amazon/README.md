## Requirements

We will be using [MiDaS](https://github.com/isl-org/MiDaS) as a baseline. In order to run it we need to install its requirements:

```
conda install pytorch torchvision opencv
pip install timm
```

We will also need tqdm, pandas, matplotlib and numpy.

## Fetching MiDaS

To fetch `midas` please use:

git submodule init
git submodule update

We will be using the `dpt_large` model, which can be found [here](https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt)

## Dataset preparation

Place the input images of the validation folder on the `input` folder of midas:

```
cp /path/to/suadd/dataset/inputs/val/*.png ./MiDaS/input
```

## Inference 

Run MiDaS with:

```
python run.py --model_type dpt_large
```

## Evaluation and visualization of results

To evaluate the results use `evaluate.py`:

```
python evaluate.py -gt /path/to/suadd_dataset_rc3/depth_annotations/val -pred /path/to/MiDaS/output -o /output/dir
```

Since the prediction it outputs is relative, this script performs an alignment step between prediction and ground-truth, least squares was used for that (see for example: [Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://arxiv.org/pdf/1907.01341v3.pdf)).

The script will print out the mean scores and save in a csv document the score of each image.

To visualize results you can use the option `--viz`, it also requires the `--input_path` pointing to the folder with the input images (i.e. `/path/to/MiDaS/input`)
