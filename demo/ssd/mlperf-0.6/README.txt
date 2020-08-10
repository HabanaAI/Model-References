Please see the other README in the sub directories too.

Notice that for SSD, you can access the coco eval package and the resnet32
backbone at:
https://console.cloud.google.com/storage/browser/mlperf_artifcats/v0.6_training

to run add current directory to the python path, download checkpoint files from gs manually.
make sure you have installed Cython and pycocotools packages and your tfrecords where generated using tf.__version__<2.
run command:
python ssd/ssd_main.py --eval_batch_size=64 --hparams=use_bfloat16=False,lr_warmup_epoch=5,base_learning_rate=3e-3,use_cocoeval_cc=true,conv0_space_to_depth=false --iterations_per_loop=625 --mode=train_and_eval --model_dir=$OUTPUT --num_epochs=64 --num_shards=1 --use_tpu=False --train_batch_size=$BATCH_SIZE --training_file_pattern=$DATASETPATH/coco_train* --use_async_checkpoint=True --val_json_file=$DATASETPATH/annotations/instances_val"$YEAR".json --validation_file_pattern=$DATASETPATH/coco_val* --resnet_checkpoint=$INIT_PATH
