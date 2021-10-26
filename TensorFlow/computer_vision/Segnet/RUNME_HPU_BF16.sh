
python3 demo_segnet.py train --train_images="dataset1/images_prepped_train/" --train_annotations="dataset1/annotations_prepped_train/" --val_images="dataset1/images_prepped_test/" --val_annotations="dataset1/annotations_prepped_test/" --n_classes=12 --input_height=320 --input_width=640 --model_name="vgg_segnet" --data_type="bf16" --epochs=125 --batch_size=16

