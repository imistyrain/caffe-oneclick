@echo off
set CAFFE_DIR=..
set eval_iter=8000
set imagepath=data/0/0.jpg
set trainedmodel=trainedmodels/platerecognition_iter_%eval_iter%.caffemodel
::set trainedmodel=platere996.caffemodel
echo %imagepath% %eval_iter%
"%CAFFE_DIR%/build/examples/cpp_classification/classification" "modeldef/deploy.prototxt" "%trainedmodel%" "modeldef/mean.binaryproto" "modeldef/labels.txt" "%imagepath%"

pause