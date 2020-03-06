# SoundNet based on HourGlass Networks

## Dependencies
- python3
- pytorch 

## Train

`python train.py
tensorboard --logdir=result/test_envelope
tensorboard --logdir=result/test_frequency
`

## Check 

`python check.py
tensorboard --logdir=result/test --samples_per_plugin images=0
`

## Evaluation
`python evaluation
`