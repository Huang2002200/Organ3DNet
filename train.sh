#!/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_AREA=2  # set the area number accordingly [1,6]

python main_instance_segmentation.py \
  general.train_mode=true \
  general.project_name="5plant" \
  general.experiment_name="area${CURR_AREA}_5plant" \
  general.filter_out_instances=true\
  general.gpus=[0]\
  data.batch_size=10 \
  data/datasets=5plant \
  general.num_targets=11 \
  data.num_labels=10 \
  trainer.max_epochs=5001 \
  general.area=${CURR_AREA} \
  data.voxel_size=0.15

