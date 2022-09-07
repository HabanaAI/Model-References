# Get COCO 2017 data sets
dir=$(pwd)
mkdir -p /data/pytorch/coco2017; cd /data/pytorch/coco2017
curl -O http://images.cocodataset.org/zips/train2017.zip; unzip train2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip; unzip val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip annotations_trainval2017.zip
cd $dir
