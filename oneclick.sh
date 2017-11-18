#!/usr/bin/env sh
set -e
TOOLS=../build/tools
DATA=data
TRAIN_DATA_ROOT=$DATA/
VAL_DATA_ROOT=$DATA/
EXAMPLE=lmdb
RESIZE=true
rm lmdb -r
mkdir lmdb
if $RESIZE;then
RESIZE_HEIGHT=20
RESIZE_WIDTH=20
else
RESIZE_HEIGHT=20
RESIZE_WIDTH=20
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    util/train.txt \
    $EXAMPLE/train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    util/val.txt \
    $EXAMPLE/val_lmdb

echo "Convert data Done."

echo "Start Training"
$TOOLS/caffe train --solver=modeldef/solver.prototxt
echo "Training Done"
