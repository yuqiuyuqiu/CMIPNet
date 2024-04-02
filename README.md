# CMIPNet
Code for paper "Lightweight Cross-Modal Information Measure and Propagation for Road Extraction from Remote Sensing Image and Trajectory/LiDAR" (TGRS, 2024)
### 1 Requirements
```
numpy
opencv_python
scikit_learn
torch
torchvision
tqdm
```
### 2 Datasets
- BJRoad: The original dataset (including satellite images and vehicle trajectories) can be requested from the author of [CVPR2019 paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Leveraging_Crowdsourced_GPS_Data_for_Road_Extraction_From_Aerial_Imagery_CVPR_2019_paper.pdf). They conduct [data augmentation](https://github.com/suniique/Leveraging-Crowdsourced-GPS-Data-for-Road-Extraction-from-Aerial-Imagery/blob/b174045888b9b7daf2c61b03fa6f922048480863/utils/data_loader.py#L57) during the training period. Here we provide the data-augmented satellite images and trajectory heatmaps (resolution: 1024*1024) used in our work. \[[Google.Drive](https://drive.google.com/file/d/1LwTn8_wpsLRBuYW7w6pmxSIhdVNGcze5/view?usp=sharing)\]   \[[BaiduYun, password：hiwv](https://pan.baidu.com/s/1kfbw0SKoQqNoG08mM-KGMA)\]

- Porto: This dataset contains 6,048 pairs of satellite images and trajectory heatmaps with a resolution of 512*512. We conduct five-fold
cross-validation experiments on this dataset. \[[Google.Drive](https://drive.google.com/file/d/1L3uqySCaIwoa-U22LTqKRemxlHhfKZL7/view?usp=sharing)\]   \[[BaiduYun, password：ffia](https://pan.baidu.com/s/1_mkVOnoTr_wxrK00t3Ac5Q)\]

- TLCGIS: This is a  road extraction dataset with 5,860 pairs of satellite images and Lidar images. Their resolution is 500*500. In this dataset, the label of foreground road is 0. \[[Download](  http://ww2.cs.fsu.edu/~parajuli/datasets/fusion_lidar_images_sigspatial18.zip)\]

### 3 Training and Testing
```bash
# experiment on BJRoad
sh train_val_test_BJRoad.sh

# experiment on Porto
sh train_val_test_Porto.sh

# experiment on TLCGIS
sh train_val_test_TLCGIS.sh
```
<!-- ## Tips -->

## 4 Training weights and prediction results

[BaiduYun, password：qktu](https://pan.baidu.com/s/1nKl_befw967y-E1tkGkS3Q)

## Citation
If you use this code, please cite our paper.
<!-- ```

``` -->


