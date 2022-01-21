1. Point $TF_MODULES_RELEASE_BUILD to Habana modules directory on your machine.
2. Run sh RUNME_HPU.sh
3. Wait for dataset to get downloaded and begin training automatically on HPU.

=======
Full logs, TF logs, and TF graph dumps: TF_CPP_MIN_VLOG_LEVEL=50 TF_CPP_MIN_LOG_LEVEL=0 TF_DUMP_GRAPH_PREFIX=my_graph_dump_dir HBN_TF_GRAPH_DUMP=2 HABANA_SEGMENT_DUMP_PREFIX=__my_prefix HABANA_LOGS=./latest_habana_logs LOG_LEVEL_ALL=0 LOG_LEVEL_GC=0 LOG_LEVEL_SYN_API=0 ENABLE_GVD=1 GRAPH_VISUALIZATION=1 python3 -m keras_segmentation train --checkpoints_path="path_to_checkpoints" --train_images="dataset1/images_prepped_train/" --train_annotations="dataset1/annotations_prepped_train/" --val_images="dataset1/images_prepped_test/" --val_annotations="dataset1/annotations_prepped_test/" --n_classes=12 --input_height=320 --input_width=640 --model_name="vgg_segnet" --steps_per_epoch=5 --epochs=1
This is similar to the command in RUNME_HPU.sh except that it adds more flags for verbosity and graph dumps and only runs a few iterations (just enough to capture logs/graphs) and doesn't train to completion

