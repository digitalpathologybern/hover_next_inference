# HoVer-NeXt Inference
HoVer-NeXt is a fast and efficient nuclei segmentation and classification pipeline. 

Supported are a variety of data formats, including all OpenSlide supported datatypes, `.npy` numpy array dumps, and common image formats such as JPEG and PNG.
If you are having trouble with using this repository, please create an issue and we will be happy to help!

For training code, please check the [hover-next training repository](https://github.com/digitalpathologybern/hover_next_train)

## Setup

Environments for train and inference are the same so if you already have set the environment up for training, you can use it for inference as well.

Otherwise: 

```bash
conda env create -f environment.yml
conda activate hovernext
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
```

or use predefined [docker/singularity container](#docker-and-apptainersingularity-container)

## Model Weights

Weights are hosted on [Zenodo](https://zenodo.org/records/10635618)

## WSI Inference

This pipeline uses OpenSlide to read images, and therefore supports all formats which are supported by OpenSlide. 
If you want to run this pipeline on custom ome.tif files, ensure that the necessary metadata such as resolution, downsampling and dimensions are available.
Before running a slide, choose [appropriate parameters for your machine](#optimizing-inference-for-your-machine)

To run a single slide:

```bash
python3 main.py \
    --input "/path-to-wsi/wsi.svs" \
    --output_root "results/" \
    --cp "lizard_convnextv2_large" \
    --tta 4 \
    --inf_workers 16 \
    --pp_tiling 10 \
    --pp_workers 16
```

To run multiple slides, specify a glob pattern such as `"/path-to-folder/*.mrxs"` or provide a list of paths as a `.txt` file.

### Slurm

if you are running on a slurm cluster you might consider separating pre and post-processing to improve GPU utilization.
Use the `--only_inference` parameter and submit another job on with the same parameters, but removing the `--only_inference`.

## NPY / Image inference

NPY and image inference works the same as WSI inference, however output files are only a ZARR array.

```bash
python3 main.py \
    --input "/path-to-file/file.npy" \
    --output_root "/results/" \
    --cp "lizard_convnextv2_large" \
    --tta 4 \
    --inf_workers 16 \
    --pp_tiling 10 \
    --pp_workers 16
```

Support for other datatypes are easy to implement. Check the NPYDataloader for reference.

## Optimizing inference for your machine:

1. WSI is on the machine or on a fast access network location
2. If you have multiple machines, e.g. CPU-only machines, you can move post-processing to that machine
3. '--tta 4' yields robust results with very high speed
4. '--inf_workers' should be set to the number of available cores
5. '--pp_workers' should be set to number of available cores -1, with '--pp_tiling' set to a low number where the machine does not run OOM. E.g. on a 16-Core machine, '--pp_workers 16 --pp_tiling 8 is good. If you are running out of memory, increase --pp_tiling.

## Using the output files for downstream analysis:

By default, the pipeline produces an instance-map, a class-lookup with centroids and a number of .tsv files to load in QuPath.
sample_analysis.ipynb shows exemplarily how to use the files.

## Docker and Apptainer/Singularity Container:

Download the singularity image from here: [TODO]

```bash
# don't forget to mount your local directory
export APPTAINER_BINDPATH="/storage"
apptainer exec --nv /path-to-container/nuc_torch_v16.sif \
    python3 /path-to-repo/main.py \
    --input "/path-to-wsi/*.svs" \
    --output_root "results/" \
	--cp "lizard_convnextv2_large" \
    --tta 4 
```
# License

This repository is licensed under GNU General Public License v3.0 (See License Info).
If you are intending to use this repository for commercial usecases, please check the licenses of all python packages referenced in the Setup section / described in the requirements.txt and environment.yml.

# Citation

If you are using this code, please cite:
```
FULL VALIDATION PAPER CURRENTLY UNDER REVIEW AT MIDL2024
```
and
```
@INPROCEEDINGS{rumberger2022panoptic,
  author={Rumberger, Josef Lorenz and Baumann, Elias and Hirsch, Peter and Janowczyk, Andrew and Zlobec, Inti and Kainmueller, Dagmar},
  booktitle={2022 IEEE International Symposium on Biomedical Imaging Challenges (ISBIC)}, 
  title={Panoptic segmentation with highly imbalanced semantic labels}, 
  year={2022},
  pages={1-4},
  doi={10.1109/ISBIC56247.2022.9854551}}
```
