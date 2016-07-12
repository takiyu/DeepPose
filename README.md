# Deep Pose #
I think more tuning is possible. If you have some advice, please tell me!

## Features ##
* `Chainer` implementation
* __Subsequent stages__
* Image viewer on web browsers. (`Flask` and `flask-socketio` are needed)

## Supported Datasets ##
- [x] FLIC
- [ ] LPS

## Testing Environments ##
### Ubuntu 14.04 ###
* Python 2.7
* Chainer 1.9.1
* OpenCV 2.4.8
* Flask 0.11.1
* Flask_SocketIO 2.4

### Arch Linux ###
* Python 3.5
* Chainer 1.9.1
* OpenCV 3.1.0
* Flask 0.10.1
* Flask_SocketIO 2.2

### Windows 7 ###
* Python 2.7
* Chainer 1.9.1
* OpenCV 3.1.0
* Flask 0.11.1
* Flask_SocketIO 2.5


## Training ##
First, download `FLIC FULL` and `FLIC PLUS` to some directory, and set the path to `settings.json`.
And also fix `CASCADE_PATHS` for your environment.

To start training, please execute the following command.

```
./scripts/train.py --stage 0
```

For subsequent stage training, `--joint_idx` argument is needed.

```
./scripts/train.py --stage 1 --joint_idx 0
./scripts/train.py --stage 1 --joint_idx 1
./scripts/train.py --stage 2 --joint_idx 0  # and so on
```
`--resume` argument is also supported.


To check current training state, please open `http://localhost:8889/` (port number can be changed by `settings.json`).
The error rate graph and visualized images can be seen.

If you want to use GPU, please set `GPU` parameter in `settings.json` to a positive number.


## Use trained models ##
Execute the following command, and open `http://localhost:8889/`.
```
./scripts/use_model.py
```
Settings is common with training (`settings.json`).

## Results ##

### First stage ###
<img src="https://raw.githubusercontent.com/takiyu/DeepPose/master/screenshots/first_stage_result.jpg">

Subsequent stages are training now.

## Async Mode Setting ##
This project uses Python `threading` or `multiprocessing` package and it can be configured by `ASYNC_MODE` in `settings.json`
On Linux `process` mode is better due to the speed, but on Windows only `thread` mode is valid.

## TODO ##
- [ ] Tune training parameters (learning rate, bounding box sigma and so on).
- [ ] Replace `multiprocess.Queue` and `Event` to `threading`'s ones on the `thread` mode.
