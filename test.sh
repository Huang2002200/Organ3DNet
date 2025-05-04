#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_AREA=2

python main_instance_segmentation.py \
general.project_name="5plant_eval" \
general.experiment_name="area${CURR_AREA}_mm5plant_test" \
general.checkpoint="" \
general.train_mode=false \
general.filter_out_instances=true \
general.save_visualizations=false \
general.export=false \
general.scores_threshold=0.0 \
data.voxel_size=0.15 \
data/datasets=5plant \
general.num_targets=11 \
data.num_labels=10 \

