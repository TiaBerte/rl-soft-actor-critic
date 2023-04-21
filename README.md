# Soft Actor Critic algorithm

This repo contains the implementation of the standard SAC algorithm and two of its variations. The [first one](https://arxiv.org/pdf/1812.05905.pdf) which add the automatic tuning of the temperature parameter $\alpha$, while the second, called [Averaged SAC](https://downloads.hindawi.com/journals/complexity/2021/6658724.pdf), computes the state value as the average of the last $K$ ones. 

The algorithms were tested on two MuJoCo Gym environments :
* [HalfCheetah-v4](https://www.gymlibrary.dev/environments/mujoco/half_cheetah/);
* [Humanoid-v4](https://www.gymlibrary.dev/environments/mujoco/humanoid/).

More information can be found in the report.

<p align="center">
<img src="https://user-images.githubusercontent.com/33131887/213513885-cf288887-ac41-48c1-aaf2-54c8cc3c851c.gif" width="300" height="300"/> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="https://user-images.githubusercontent.com/33131887/213512242-33cba048-89f2-41a7-be7e-4f1e35adc398.gif" width="300" height="300"/>
</p>

## How to use 
The main is an .ipynb file. It allows you to train the model in Colab using the provided free GPU. It is sufficient to pass to the parser the desired arguments.  
The default arguments are the ones used for my experiments.

```
--training     : flag for training the model.
--test         : flag for testing a model.
--env_name     : environment to use. 
--model_path   : path from which to load a model, if None the training start from scratch.
--alpha_tuning : flag for selecting the variant with automatic tuning of α parameter.
--K            : number of values to average for computing the state value. If different from 0 it
                 automatically select the Averaged-SAC variant.
```





## Repo structure
<em>checkpoints/</em> contains the last best checkpoint for each version of the algorithm for both the environments.   
<em>buffer/</em> contains the buffer related to the last best checkpoints (Humanoid buffers were too big to be uploaded.).
https://user-images.githubusercontent.com/33131887/233627279-234e7d7d-7b5c-4eb3-adf1-6b1e7c09527a.mp4
