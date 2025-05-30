# HistoCell<img src="image/README/logo.png" width = 150 align = right>

**HistoCell** is a **weakly-supervised deep learning framework** to elucidate the **hierarchical spatial cellular information** including **tissue compartments, single cell types and cell states** with **histopathology images only**. This tutorial implements HistoCell to predict super-resolution spatial cellular information and illustrates the representative applications. The link to the HistoCell method will be presented soon. \
Our website: http://histocell.qhdyr.net/index/index/index.html

<img src="image/README/Intro.jpg" alt="Image" width="800" style="display: block; margin: 0 auto;">

## Environments
```sh
pip install -r requirements.txt
```

## Data Format and Preprocessing

### Data Preparation

* For model pre-train, HistoCell takes the **scRNA-seq** and **spatial transcriptomics** data with paired **high-resolution histopathology images** as input.
  * For histopathology images, we cut the paired images according to the pixel coordinates from ST data. The preprocessing code can be found in **./tutorial/tutorial.ipynb**
  * For transcriptomics data, we apply deconvolution methods to get the cell composition as the supervision. Applicable methods are listed as [CARD](https://www.nature.com/articles/s41587-022-01273-7), [RCTD](https://www.nature.com/articles/s41587-021-00830-w), [Tangram](https://www.nature.com/articles/s41592-021-01264-7), [Cell2location](https://www.nature.com/articles/s41587-021-01139-4) etc.
* For model inference, HistoCell only requires the histopathology images including tiles and WSIs.
  * WSIs should be formatted as .svs or .tif. In order to convert the WSIs into the tiles can be processed in a high-throughput manner, we apply the toolbox from [CLAM](https://github.com/mahmoodlab/CLAM) for image segmentation, stitching and patching. As for TCGA diagnostic images, patch of 256x256 pixels is recommended.
  * Tiles can be storaged in any format of image.

### Data Preprocessing

Using GPUs is highly recommended for whole slide images (WSIs) processing.

Before we begin the model inference process, cell segmentation is required. Here, we apply [HoVerNet](https://pdf.sciencedirectassets.com/272154/1-s2.0-S1361841519X00079/1-s2.0-S1361841519301045/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEO3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDd2OMed8Quier79h6hhXShEPV1a3lwQv%2BMd%2B99MpGolQIgHbvYbhoiHEe1uT1QjfLGEOEMBSTSCAhJmoThBfbNll4quwUIpv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDMf%2B5OgldA9yxfGLPyqPBY%2FB1K2p3P7IvTS1hGh8akwWej0tXrgiaex2nNOJT6jwHbFCml3oXhsoNvr5bQ%2BmhFL3hNeKRbOXowl1RfkkrM9Mo8W7VB6L6a3faDuy3R9FmYg9OniS%2F2l1pasqKf%2Bk3es0ZtkBZYJhIpRcxWIogcRB1WPWE9WGuBRfN2qp7xf7NNkq4ZmbNaU3ysqx%2FMFFZWGP1DoLhVUeP18olpZstHpJ5rrKvMEJ4bUhOnN4WkA4wflhpJAKy6dv10PJIbCGYWReuhcTFO%2FNoSqCRDUUnQZD5zRfaCfsjNO943WEECuHreEcGSfsGwH16ncrE6deBpvS9x5f7qFSzLkM01th0ZwonFL0zXSGN6qaPnZ0wBzO2Lbe0OtzPBeHG3BrPl3VxL9qYSKNDITFNW%2BVRAO3CckWm%2Bt%2FEQqBGRbX%2FArLKvT7NS12jx%2FhEhj%2B%2Ba3yYQQjFUJMDPfLpbXsLpl8IOsNKBbQAqsT09iN0an0zA8q7oh%2F9HgfL8KCZdRuPWp9HkLLjSRbdH01i7ctSMTbehrkjiVMnXz8f3B9%2BVHfZR%2B3xQYH2YOH67UE87JCofjKJkWhroXKSkS1c53ye%2FOCEyF9gp2ChrWnKG8o95jTccF%2BInoECXr0Ymc5QiotpLF2es6pUQcGi2mq5rcjY1P6vJ9x4i4DDR2e%2FM718BZaM8zQCUmYm4XOBu%2B22Wtf8GgAymn8pz7uTgKnh9jlZhmTZ4YAnzVaqW%2FjTovCSRCUucEzklibwjVJFHw8urGagWh77nG0Qv9wGnO4PQYYOTJqs%2F2WZo9raeKVwxMPkcoXOVRM0Pkphg4bDnVZG63R1xL89urfYK1PPKDGtH08o9UvREJM6ugGTjDHpRZPeS29NYXNZ7kwxtHOrQY6sQFFrXdmd2FpmYFJtlfg2DJxJa7SwFYBVdj3Db6HD%2FLftOy%2B%2FyGCiRWyB%2FC%2FDxPPU0WDWvUzAq72HoaEV87cVsLz6Q7446UGQ1HAGMKRnAhCALhSZIp7WfyC1gTuAthi5QVJvr78GXQAinqnBaAlrMaHnLTtbiSNPNykRQrrEvhBDIq21Ffy%2Fid%2F9pPW0IURXrvj1end6m7dZT7ZtEIYcA%2BWVa5%2FYz6%2FOu5Tvta6zaDkE0k%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240126T134143Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY5OQRHCPI%2F20240126%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=608d9d59c577a23f89bb7aaa69c7f8c2163e3911b0cfcdc16987fb1603f50428&hash=48169297d0c9fa366d3cb8bd120add89965ee3917df478d735f708fb300c8168&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1361841519301045&tid=spdf-4ba996cc-271e-4fff-93e1-559385625235&sid=31ab0cf595ec25414989848882e6352d1712gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f135a56045e56020451&rr=84b92cd0cc162287&cc=us)[[code](https://github.com/vqdang/hover_net)], a well-developed nuclei instance segmentation method, to process cell segmentation from image tiles. It is worth mentioned that other segmentation methods (e.g. [MaskRCNN](https://arxiv.org/abs/1703.06870), [Cerbebrus](https://pdf.sciencedirectassets.com/272154/1-s2.0-S1361841522X00078/1-s2.0-S1361841522003139/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEO3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDWnVS2nFNswtWtc37yxAthvAru8F%2Bi9sXObOpOSp4fcQIgPLxoDxjXB2%2Bn5LDajvayiIU5Ev5%2FZSwXH65%2FSCvHpTQqvAUIpf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDGqoAHUKPFM7LE85KCqQBRI1MjQTxc06bBJ2fe8tBSgL1PqcV3eXn3SDFwCsfPX48S61JVtwG9g91MxurBEUNhdwrjWdNBwI3pDUNKp7MQIH0AJVRXhwJ5mKktV96eZd5dizuGWGAicjTD%2Fy5zusW3nLiEhx0ka61bVUzKU7ZGdA%2Bs%2BT1egxuN3LK9fHKGVF63NVJt2sbDgNigMWEEiPdlXbWtvqvB0EylrEcVT7Rqo2a7ccsmPMCyDQlKyMzWJpNBOOu1KI2BK2KE0y0jLELSTeOvF9iHfwmBOf3Q1Azq7uBElKYA2jQPPFm6O%2Fi7b6UsJQtg5RLxB2HbOcRK1htkMmnOvGARy1y%2BbwQ26nYJJW5C85m%2F%2Bc3%2FXjBg%2BrsLeodZYEWQEpVgY%2BTUf%2Bdi4%2BCYmok0ra8kTrShb2CvgVK4pVrk8L0AO33WVGNokeOp%2F8xV0bz7O1DAsqN2ykAvvPk0wuprohZxBm2BGhA%2BLY7lox3f6v6Rv544wpvF8RHreBS%2B2jkVPae8%2FZER6WaRl9WuFuBzJlgY0xdW1V2iU8quvRxltyx%2BSS%2BVbdgMMBmX9H41LLVQgTYNnciL7hTcHRzS70pXQa2fb%2BfhFjQ0dXv3eH%2BCz3JeKgpU6wInT5Ax4GKKU9o3QqJMh9OsFmMvKjVHL8bzhpllg77aZEVPHbXj%2F2oUB6xN3BhSjl30wKiDKT0y1RGt2QuKWOqd%2FW5nEo0Nw8YA4kS%2FIkq6v7XQuyJnuPx6orceBGrHbBb0mLvHV5XubTTAmmvGmXojPqBT6cW%2FmCdjwHnGoAb5HG%2BAPuc%2FwCJ8pDR9ZxDgzry%2F4TNCPICSIP7yj%2B9tyfleO3i1J1mhnzCa%2BINadYcFBBJcVN4PW6abUmLq2zakWegZOr%2FjW2MNrHzq0GOrEBslcHzW1k%2BVNCvsQQ9R4j1edYgD16Ya5im3eeb4DyfkHwxF1eODT4NU1SsSI3uXG%2BnfmMEP0qPsq6khy%2FK9RXojWq0Qn0lsTw9ilEvPN97m192kvnNcD0dkbN32Jpjl%2Bfg08nRzORU9wzjJfwvSjwx2RIMdnBtJGENnzNNt%2FuEg%2BWdHon%2F6%2FG3%2FvmkWweeuQw9hnbO6X2HV0BGyMPXcMq9ouE8eIDqG38lbE3hm9kUFm1&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240126T134736Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY72FFPE6G%2F20240126%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=f81b493864396df90b2dc5a4bc19a720f03415ca06d0e42870d9bbc34adb93ef&hash=e4bf9d21b009b09db822df8fbcdc06443d696a664e097e20f414559b75f39f7d&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1361841522003139&tid=spdf-1b0d3161-dd23-45aa-bfe3-aa725b637c50&sid=31ab0cf595ec25414989848882e6352d1712gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f135a56045e57545753&rr=84b935728ba1239d&cc=us)etc.) are also applicable here.

With cell segmentation by HoVerNet, you will get json file as the segmentation results illustrated as below:

```json
{"mag": null, "nuc": {"YOUR_CELL_ID": {"bbox": [[1, 2], [3, 4]], "centroid": [2, 3], "contour": [[0, 1]], "type_prob": 0.9, "type": 0}}}
```

### Input data format
The tiled images, segmentation results(json file) and cell type/state proportion given by deconvolution are required during training process.

## Data Accession
Demo data and tutorial data could be assessed on the Google drive with [Demo](https://drive.google.com/file/d/1DzwsFvD4wWKNN1wJCe3YsQZlk9xIp8DV/view?usp=sharing) and [Tutorial](https://drive.google.com/file/d/1iwXzxafdUF7SF6IAEyWluN728WbxAMEi/view?usp=sharing), respectively. To run the demo, you need to download the data to the directory **./demo/data**.

## Model pre-training with ST data
Run **train.py** for model pre-training on cell type prediction. You need to change the parameters in **configs.py** including the model information, dataset directory and training details. Besides, choose the proper tissue compartment file in **./tcs**. To develop your own model, run the command below.
```python
python train.py \
--model YOUR_MODEL_DESC \
--tissue TISSUE_TYPE \
--deconv DECONVOLUTION_METHOD \
--prefix SAMPLE_PREFIX \
--k_class NUM_OF_CLASS \
--tissue_compartment PATH_TO_TCS_FILE
```
A simple demo on subject H1 from HER2ST dataset:
```python
python train.py \
--model Breast_Benchmark_H1 \
--tissue BRCA \
--deconv RCTD \
--prefix H1 \
--k_class 6 \
--tissue_compartment ./tcs/tissue_compartment_BRCA.json
```

For the benchmark analysis on the PanNuke dataset, run the leave-one-out cross-validation command:
```
python train_oneout_pannuke.py \
--model PanNuke_cross_validation \
--tissue Breast \
--reso 1 \
--deconv Mix \
--prefix fold1 fold2 fold3 \
--k_class 5 \
--tissue_compartment ./tcs/tissue_compartment_Mix.json
```

As for cell state, you can run **train_state.py** to develop the cell state prediction model. For example, develop the cell state prediction on the breast cancer ST samples:
```python
python train_state.py \
--model brca_state \
--tissue BRCA \
--deconv RCTD \
--prefix 10x_BRCA A_1938345_11 B_1938529_9 C_2000752_23 D_2000910_33 \
--tissue_compartment ./tcs/tissue_compartment_state.json
```

## Quick inference with histopathology images only
With the pretrained model, you can infer the cellular spatial profile with **infer.py**.
```python
python infer.py \
--model Breast_Benchmark_H1 \
--epoch 30 \
--tissue BRCA \
--deconv RCTD \
--prefix H1_hires \
--k_class 6 \
--tissue_compartment ./tcs/tissue_compartment_BRCA.json \
--omit_gt
```

## Representative Results
Here we only illustrate the demo and representative results corrsponding to the paper in the tutorial.ipynb. The predicted hierarchical spatial cellular information is storaged as a dict in a pickle file for each slide. For more results, you can directly jump to our [HistoCell website](http://histocell.qhdyr.net/index/index/index.html).

### Benchmark results

* **Tissue Compartment**
  <div align = center>
  <img src="image/README/tissue_compartment.jpg" alt="Image" width="400" align="center">
  </div>
* **Single-cell Type**
<div align = center>
  <img src="image/README/cell_type2.jpg" alt="Image" width="500" style="display: block; margin: auto auto;">
</div>
<div align = center>
  <img src="image/README/cell_type1.jpg" alt="Image" width="500" style="display: block; margin: auto auto;">
</div>

  Red, blue and green scatters represent cancer epithelial cells, stromal cells and macrophage cells.
* **Cell States**
<div align = center>
  <img src="image/README/cell_state.jpg" alt="Image" width="800" style="display: block; margin: auto auto;">
</div>

### Representative Application: Tissue architecture annotations

With a histopathology image, HistoCell could first infer pixel-level cell types and then cluster cells as tissue regions, which exhibit high accuracy and allow users to further identify the small foci within tissue regions at pixel-level resolution. 
<div align = center>
<img src="image/README/segmentation.jpg" alt="Image" width="500"  style="display: block; margin: auto auto;">
</div>

### Representative Application: Cell Type Deconvolution

Since HistoCell Integrates spot-level cellular compositions deconvoluted from expression data and those based on histologic morphologic features, it could produce a more precise and robust deconvolution result. 
<div align = center>
<img src="image/README/deconvolution1.jpg" alt="Image" width="500" style="display: block; margin: auto auto;">
</div>
<div align = center>
<img src="image/README/deconvolution2.jpg" alt="Image" width="500" style="display: block; margin: auto auto;">
</div>

### Representative Application: Spatial organization indicators identification
<div align = center>
<img src="image/README/SOI.jpg" alt="Image" width="400" style="display: block; margin: auto auto;">
</div>
The histopathology image is coverted to **spatial cellular map** with HistoCell and the cells are accumulated as clusters. Through the correlation analysis between clinical outcomes and cellular spatial clustering features, we identify spatial biomarkers for prognosis. Demo results can be found in Tutorial.ipynb. The representative spatial features for prognosis stratification is visualized as below.
<div align = center>
<img src="image/README/biomarker.jpg" alt="Image" width="1000" style="display: block; margin: auto auto;">
</div>




