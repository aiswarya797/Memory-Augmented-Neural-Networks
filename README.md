### Memory-Augmented-Neural-Networks
This repository contains code and analysis of memory augmented neural networks.
This is a tensorflow implementation of \href{https://github.com/tristandeleu}{tristandeleu's} excellent Theano implementation.
The code is inspired by \href{https://github.com/vineetjain96}{Vineet Jain}.

### Usage 
python MANN.py --batch-size 16 --num-classes 5  --num-samples 50 --input-height 20 --input-width 20 --num-reads 4 --controller-size 200 --memory-locations 128 --memory-word-size 40 --learning-rate 1e-4 --iterations 100000 --path '/media/aiswarya/New Volume/My_works/ISI_contd/images_background'

### Data
Unzip images_background.zip from \href{https://github.com/brendenlake/omniglot/tree/master/python}{Data} and add the path to 'path' argument for running the code.

