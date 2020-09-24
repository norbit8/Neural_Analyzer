# Neural Analyzer
A python package that includes methods for decoding neural activity.
Specifically used in the expirement we sample neural activity from the [Basal ganglia](https://en.wikipedia.org/wiki/Basal_ganglia) and the [cerebellum](https://en.wikipedia.org/wiki/Cerebellum) (Using a *Macaca fascicularis* monkeys) while the monkeys are targeting moving rectangles.

In the following project we build a decoder where given a vector of spikes of neural activity we decode the direction of the eye.

All of the work is based on machine learning algorithms (specifically KNN) which we train using the given data.
First, we suggest you to use `conda` in order to reconstruct our work environemnt. 
To recreate the environment you can do the following:
   ```conda env create -f environment.yml```
