# Generating-TV-Script

This project will generate [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.  Part of the [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts from 27 seasons are used in this project.  The Neural Network will generate a new TV script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern).

### Results

Below are the generated scripts:
'''
moe_szyslak:(sighs) what's the point?... same ol' stinkin' world...(to moe) he seems nice.
lisa_simpson: how'd mindless tester's off my this friend is gonna go tsk, tsk, tsk, tsk, tsk, tsk, tsk till broom.
carl_carlson: i thought you said chug-monkeys. what beverage, brewed since ancient times, is made from hops and grains?
scary him miss_lois_pennycandy:.
lenny_leonard: plus his wife was madonna.
ned_flanders: what're blessing.(small sob)
moe_szyslak: eh, sam: call for the musical in the pledge of allegiance. bugging me.


moe_szyslak: hates me, sam: laugh at the musical in the fridge, the man who spews harmony) pope's beers, you just gotta warn you, they must be the ugliest beer and a wad of bills.


moe_szyslak: hey, hey, hey, hey! freaky.


moe_szyslak:(dumbest glass, sam:) dad?
scary him on, sam: lee is walther hotenhoffer and i'm in the pharmaceutical
'''

### The Files

* 'check_tensorflow_gpu.py' : check tensorflow version and whether your machine have a gpu available.
* 'preprocess.py' : methods to preprocess, save, load data and parameters.
* 'build_network.py' : set up the structure for RNN and put it all together by building a tensorflow graph
* 'train_network.py' : train the model using the structure built in 'build_network.py'
* 'generate_script.py' : utilize the trained model to generate TV scripts
* 'main.py' : pass parameters, variables and data to the model, 'main.py' makes the model much easier to train and to tune.
