
**Pytorch implementation of 3D SOC-Net: Deep 3D Reconstruction Network based on Self-Organizing Clustering Mapping.**

![Overall Architecture](./img/Architecture.png)

## Datasets
We use the [ShapeNetCoreV1 dataset](https://www.shapenet.org/) in our experiments, the ShapeNet rendering images are from http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz

Our prepared dataset is available [here](https://xmueducn-my.sharepoint.com/:f:/g/personal/dmt1809254_xmu_edu_my/EvWZ-eu9RbJJothbOV1C8uQBOCCjTY5_awhV0ubt0HtrJw?e=WyTEeg)

For executing the project, only all the pkl files are needed. The pkl files should be put inside dataset_split folder.

## Pretrained Models

The pretrained models on ShapeNet are available [here](https://xmueducn-my.sharepoint.com/:f:/g/personal/dmt1809254_xmu_edu_my/EgBCkDNkVhlMtJqZ_DvExW8Bu9DTOJ8ZzYTVi9avn-mV9g?e=4dnhmM). Stored in the corresponding output/best folder.

## Executing Environment
The code has been tested on Windows 10, Python 3.7.9, PyTorch 1.8.0, CUDA 10.2

## Get Started
To train all the models, you can simply execute main.py in execute folders.
To train a specific model, you can execute main.py in corresponding folders.

## Testing Results
The table below shows the quantitatlive testing results. Note that the CD<sub>1</sub> and EMD<sub>1</sub> refer to results obtained from the original works, whereas CD<sub>2</sub> and EMD<sub>2</sub> denote the replicated results.
![Quantitative Testing Results](./img/QuantitativeResult.png)

The table below shows the visualization of testing results.
![Qualitative Testing Results](./img/QualitativeResult.png)
