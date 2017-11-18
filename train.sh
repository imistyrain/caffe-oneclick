#!/usr/bin/env sh
set -e
RESUME=1
RESUME_ITER=10000
TOOLS=../build/tools
resumemodel="platerecognition_iter_$RESUME_ITER.caffemodel"
if [ $RESUME -eq 1 ]; then
echo "Resume from $resumemodel"
$TOOLS/caffe train --solver=modeldef/solver.prototxt --weights="trainedmodels/$resumemodel" $@
else
echo "Start Training"
$TOOLS/caffe train --solver=modeldef/solver.prototxt $@
fi
echo "Training Done"
