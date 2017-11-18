@echo off
set CAFFE_DIR=..
set eval_iter=10000
set imagepath=data/0/0.jpg
set trainedmodel=snapshot/plate_iter_%eval_iter%.caffemodel
::set trainedmodel=platere996.caffemodel
echo %imagepath% %eval_iter%
"%CAFFE_DIR%/build/examples/cpp_classification/classification" "models/deploy.prototxt" "%trainedmodel%" "models/mean.binaryproto" "models/labels.txt" "%imagepath%"
pause