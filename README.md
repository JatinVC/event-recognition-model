# event-recognition-model
Pull the code using:
`git pull https://github.com/JatinVC/event-recognition-model.git`

create a python environment using conda command:
`python -m venv snnenv`

activate the environment by doing:
* navigate into the snnenv folder, and locate the activate script
* run the activate script and you should be in the environment

install all requirements by doing:
`pip install --r requirements.txt`

deactivate the environment by doing:
`deactivate`

### Writeup for the group
basically the issue with the snn right now is that the data it is taking in is being upscaled to the point where each image 
it creates requires about 1gb of RAM to actually process.

This issue comes into account at around line 138 where it goes:
`loss = neuron.Tempotron.mse_loss(v_max, net.tempotron.v_threshold, label.to(device), 10)`

I think the matrix shapes are fine so that shoulnd't be the issue,

two possible solutions
- resize the tensor to be of shape 128, 128 and reset the neural network to work with it (ill give the code for the neural net)
- somehow change the code in the loss function so it doesn't take as much RAM as before.

### What this code is doing
- we are loading in the data from the raw files and converting them into frames (look into __init__.py inside the dataset folder to see this)
- and passing it into the neural network to train it.
- We are bringing in the dataset and doing preprocessing to convert it to frames using some method (this im not sure at all how it works)
