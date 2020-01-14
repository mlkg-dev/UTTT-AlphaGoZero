# UTTT-AlphaGoZero
An implementation of the AlphaGo Zero algorithm for Ultimate tic-tac-toe.
This is a successful attempt to adapt the algorithm to the different game.

## Goals of the project
1. Reproduce DeepMind results and prove that their algorithm works for various types of games.
2. Test the effects of different values of hyperparameters on model learning.

## Getting started

### Requirements
* Python 3
Libraries:
* NumPy
* Keras
* Tensorflow

You can install them by running this script:
```
pip3 install -r requirements.txt
```

The project should run with the newest version of the above libraries, but if it's not the case try this working configuration:
* NumPy 1.17.3
* Keras 2.3.1
* Tensorflow 2.0.0

If you want to use GPU install tensorflow-gpu and [its requirements](https://www.tensorflow.org/install/gpu) instead.

### Usage
This project assumes that you have at least general idea how AGZ algorithm works. [Here](https://adspassets.blob.core.windows.net/website/content/alpha_go_zero_cheat_sheet.png) is a great summary.

The training pipeline consists of three stages:
1. Self-play
2. Network training
3. Testing network.

#### Self-play
Run
```
python3 Selfplay_module.py
```
After generating a large number of games (around 40k) go to the next step.

#### Network training
Run
```
python3 Training_module.py
```
The network after training is saved as "new_network.h5".

#### Testing network
Run
```
python3 Test_module.py <old_network_name> <new_network_name>
```
You should check if the network after training performs better than the previous one. The pass rate is 50%. If the network fails retrain it or generate more games. After passing rename the network as "current_network.h5" and you are ready to repeat the three stages again.

## Results
Due to smaller computing capabilities than DeepMind's supercomputers, I only trained one new generation of network. It passed the test with 62% wins.

## FAQ
In progress...

## TODO
* Add multithreading to MCTS
* Allow saving generated games in JSON
* Automate the training process
* Optimization
* Bugfixing
* Code cleaning

## Related links
[About Ultimate tic-tac-toe](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe)

[AlphaGo Zero paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)

