# hover_next_inference
Inference code for HoVer-NeXt

Pre-print:



## Setup

Environments for train and inference are the same so if you already have set the environment up for training, you can use it for inference as well.

Otherwise: 

```bash
conda env create -f environment.yml
conda activate hovernext
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
```

or use predefined [docker/singularity container](#docker-and-apptainersingularity-container)

## WSI Inference

This pipeline uses OpenSlide to read images, and therefore supports all formats which are supported by OpenSlide. If you want to run this pipeline on custom ome.tif files, ensure that the necessary metadata such as resolution, downsampling and dimensions are available.

Before running a slide, choose [appropriate parameters for your machine](#optimizing-inference-for-your-machine)

To run a single slide:

```bash
python3 main.py \
    --input "/path-to-wsi/wsi.svs" \
    --output_root "results/" \
    --cp "lizard_convnextv2_large" \
    --tta 4 \
    --inf_workers 4 \
    --pp_tiling 8 \
    --pp_workers 4
```

To run multiple slides

### Slurm

if you are running on a slurm cluster you might consider separating pre and post-processing to improve GPU utilization.

## NPY inference

NPY Inference works the same as WSI inference, however output files are only a ZARR array

```bash
python3 main.py \
    --input "/path-to-wsi/wsi.npy" \
    --output_root "/results/" \
    --cp "lizard_convnextv2_large" \
    --tta 4 \
    --inf_workers 4 \
    --pp_tiling 8 \
    --pp_workers 4
```

Support for other datatypes are easy to implement. Check the NPYDataloader for reference.

## Optimizing inference for your machine:

1. WSI is on the machine or on a fast access network location
2. If you have multiple machines, e.g. CPU-only machines, you can move post-processing to that machine
3. '--tta 4' yields robust results with very high speed
4. '--inf_workers' should be set to the number of available cores
5. '--pp_workers' should be set to number of available cores -1, with '--pp_tiling' set to a low number where the machine does not run OOM. E.g. on a 16-Core machine, '--pp_workers 16 --pp_tiling 8 is good

## Using the output files for downstream analysis:

By default, the pipeline produces an instance-map, a class-lookup with centroids and a number of .tsv files 


## Docker and Apptainer/Singularity Container:

```bash
# don't forget to mount your local directory
export APPTAINER_BINDPATH="/storage:/storage"
apptainer exec --nv /path-to-container/nuc_torch_v16.sif \
    python3 /path-to-repo/main.py \
    -p "/path-to-wsi/wsi.svs" \
    -o "/results/" \
	--cp "convnextv2_large_focal_fulldata_0" \
    -tta 4 \
    --slurm


```

# Cite

If you are using this code, please cite:

TODO
