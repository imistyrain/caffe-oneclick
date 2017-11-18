#!/usr/bin/env sh
set -e
DATA=data
TOOLS=../build/tools
regeneratelmdb=0

convert_lmdb(){
    RESIZE_HEIGHT=20
    RESIZE_WIDTH=20
    echo "Creating train lmdb..."
    GLOG_logtostderr=1 $TOOLS/convert_imageset \
        --resize_height=$RESIZE_HEIGHT \
        --resize_width=$RESIZE_WIDTH \
        --shuffle $DATA/ \
        util/train.txt \
        lmdb/train_lmdb
    echo "Creating val lmdb..."
    GLOG_logtostderr=1 $TOOLS/convert_imageset \
        --resize_height=$RESIZE_HEIGHT \
        --resize_width=$RESIZE_WIDTH \
        --shuffle $DATA/ \
        util/val.txt \
        lmdb/val_lmdb
}
create_lmdb(){
    if [ -d lmdb ] ; then
        if [ $regeneratelmdb -eq 1 ] ; then
            rm lmdb -rf
            mkdir lmdb
            convert_lmdb
        fi
    else
        mkdir lmdb
        convert_lmdb
    fi
}
train(){
    if [ ! -d output ]; then
        mkdir output
    fi
    latest=$(ls -t models/*.caffemodel | head -n 1)
    if [ -f ${latest} ]; then
        echo "Resume training from ${latest}"
        $TOOLS/caffe train --solver=models/solver.prototxt --weights=$latest
    else
        echo "Start Training"
        $TOOLS/caffe train --solver=models/solver.prototxt
    fi
    echo "Done"
}
# python evaluate
evaluate(){
    latest=$(ls -t output/*.caffemodel | head -n 1)
    echo "Evaluating "${latest}
    python util/evaluation.py --weights=${latest}
    cd ..
}
# c++ evaluate
run(){
    if [ not -d build ]; then 
        mkdir build
    fi
    cd build
    cmake ..
    make -j8
    ./evaluation
    #./cpp4caffe
}
# generate util/train.txt and val.txt for training
# python3 util/preprocess.py
#create_lmdb
# train
python train.py
evaluate
#run