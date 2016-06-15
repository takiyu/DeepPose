# Deep Pose #
This project needs more tuning. If you have some advice, please tell me!

## Features ##
* `Chainer` implementation.
* Subsequent stages.
* Image viewer on your web browser. (`Flask` and `flask-socketio` are needed)

## Supported Datasets ##
- [x] FLIC
- [ ] LPS

## Training ##
First, download `FLIC FULL` and `FLIC PLUS` to some directory, and set the path to `settings.json`.

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


To check current training state, please open `http://localhost:8889/`.
The error rate graph and visualized images can be seen.

If you want to use GPU, please set `GPU` parameter in `settings.json` to a positive number.


## Use trained models ##
Execute the following command, and open `http://localhost:8889/`.
```
./scripts/use_model.py
```
Settings is common with training ('settings.json').

## Results ##

### First stage ###
<img src="https://raw.githubusercontent.com/takiyu/DeepPose/master/screenshots/first_stage_result.jpg">

Subsequent stages is training now.
