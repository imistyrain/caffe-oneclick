@echo off
set CAFFE_DIR=..
set DATA=data
set REISZE_DIM=20
set converttool=%CAFFE_DIR%/build/tools/convert_imageset
set finetunemodel=plate996.caffemodel
if exist "lmdb/train_lmdb" (set regenlmdb=0) else (set regenlmdb=1)
::set regenlmdb=1

if %regenlmdb% equ 1 goto regeneratelmdb

goto train
:regeneratelmdb
echo "Creating train lmdb..."
del "lmdb/train_lmdb\*.*" /f /s /q
del "lmdb/val_lmdb\*.*" /f /s /q
rd /s /q "lmdb/train_lmdb"
rd /s /q "lmdb/val_lmdb"
rd /s /q lmdb
mkdir lmdb
"%converttool%" --resize_height=%REISZE_DIM% --resize_width=%REISZE_DIM% --shuffle "%DATA%/" "util/train.txt" "lmdb/train_lmdb"
echo "Creating val lmdb..."
"%converttool%" --resize_height=%REISZE_DIM% --resize_width=%REISZE_DIM% --shuffle "%DATA%/" "util/val.txt" "lmdb/val_lmdb"

echo "Computing mean:"
"%CAFFE_DIR%/build/tools/compute_image_mean" "lmdb/train_lmdb" "modeldef/mean.binaryproto"

:train
if exist %finetunemodel% (set extra_cmd="--weights=%finetunemodel%")
"%CAFFE_DIR%/build/tools/caffe" train --solver=modeldef/solver.prototxt %extra_cmd% 
echo "Done"
pause