# Image Servers #
Image server modules for development of deep learning or image processing.

Each module is written with Flask, Flask-Socketio and ReactJS,
so client browser will be notified of some changes immediately.

## Dependences ##
* Python2 (or 3)
* Flask
* Flask-Socketio
* OpenCV

## DNN Examples ##
To run DNN examples, please install chainer and
download `bvlc_googlenet.caffemodel` to `dnn_models` (see `dnn.py`).

## Screenshots ##
### Viewer ###
<img src="https://raw.githubusercontent.com/takiyu/image_servers/master/screenshots/viewer1.png" width="400px">
<img src="https://raw.githubusercontent.com/takiyu/image_servers/master/screenshots/viewer2.png" width="400px">

Visualize each CNN layer

### Uploader ###
<img src="https://raw.githubusercontent.com/takiyu/image_servers/master/screenshots/uploader1.png" width="400px">

Capture client webcam, upload to server and predict it (the upper picture is `clock`).
