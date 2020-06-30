### Memory-Augmented-Neural-Networks
This repository contains code and analysis of memory augmented neural networks.
This is a tensorflow implementation of https://github.com/tristandeleu excellent Theano implementation.
The code is inspired by https://github.com/vineetjain96.

### Usage 
python MANN.py --batch-size 16 --num-classes 5  --num-samples 50 --input-height 20 --input-width 20 --num-reads 4 --controller-size 200 --memory-locations 128 --memory-word-size 40 --learning-rate 1e-4 --iterations 100000 --path '/home/aiswarya/My_works/MANN/images_background'

### Data
Unzip images_background.zip from https://github.com/brendenlake/omniglot/tree/master/python and add the path to 'path' argument for running the code.

