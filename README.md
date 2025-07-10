# the code for our WACV2025 paper "Cascaded Dual Vision Transformer for Accurate Facial Landmark Detection"


# train
download the pre-cropped WFLW images from https://github.com/starhiking/HeatmapInHeatmap?tab=readme-ov-file, unzip the WFLW.zip under root folder and run the following commands:
        
        torchrun --nproc_per_node=2 TrainHeatmapStageFP16.py

