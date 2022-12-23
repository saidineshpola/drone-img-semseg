## Requirements

Testing was done with Python 3.9. The following python modules are needed:

```
fire
tqdm
pytorch
torchvision
matplotlib
prettytable
```

## Fetching MMSeg

Requirements of MMSeg include mmcv-full. It can be installed with:
```
pip install -U openmim
mim install mmcv-full
```

Then fetch MMSeg:
```
git submodule init
git submodule update
```

## Apply SUADD '23 patch

After fetching MMSeg we need to apply the provided patch to add the dataset and configuration for SUADD '23:

```
cd mmsegmentation
git apply ../mmseg_suadd.patch
```

## Dataset preparation

<!-- We will be modifying the dataset in place, so it is recommended to create a backup before continuing.

For this model we will modify slightly the labels, since we do not want to predict the background.

The script `scripts/convert_labels.py` can be used to convert the labels of the dataset as:
`python script/convert_labels.py --annotation_dir path/to/annotations/`. -->

Create a data directory within `mmsegmentation` and link the dataset:

```
mkdir mmsegmentation/data
ln -s /path/to/dataset mmsegmentation/data/suadd23
```

## Training

You can run the training with `mmsegmentation/tools/dist_train.sh`. For example to train using 1 GPU:

```
cd mmsegmentation
./tools/dist_train.sh ./configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_suadd.py 1 --deterministic
```

The `--deterministic` flag will slow down training but create reproducible results.

By default the output will be printed to the folder `mmsegmentation/work_dirs`.

## Evaluation

To retrieve the mean intersection over union (mIoU) score, run:

```
./tools/dist_test.sh ./configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_suadd.py ./work_dirs/segformer_mit-b5_8x1_1024x1024_160k_suadd/latest.pth 1 --out ./work_dirs/segformer_mit-b5_8x1_1024x1024_160k_suadd/eval.pkl --eval mIoU
```

To visualize results you can save the output of the segmentation in RGB format:
```
./tools/dist_test.sh ./configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_suadd.py ./work_dirs/segformer_mit-b5_8x1_1024x1024_160k_suadd/latest.pth 1 --format-only --eval-options "imgfile_prefix=./suadd_test_results"
```

This command will store a visual representation of the segmentations on the `./suadd_test_results` folder. To inspect visually how well your moder perform, you can use the script `visualize_results.py` to overlap the images with the input images:

```
python visualize_results.py /path/to/inputs /path/to/rgb/labels /path/to/output/folder
```

To compare with the ground truth:
```
python visualize_results.py /path/to/inputs /path/to/rgb/labels /path/to/output/folder --gt_path /path/to/gt --debug
```