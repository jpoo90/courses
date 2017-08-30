# Deep learning

## Applications
+ Analizing crops. 
+ Analizing medical images.
+ Hedge funds analisis. 
+ Fashion. 

Universal learning machine, requires and we have right now: 
- Infinite flexible function 
  Neuronal network is the function. They are universal approximation machines. It's capable on handling any mathematical function we give them.
- All-purpose parameter fitting. 
  Gradient descent, and backward propagation. 
- Fast and scalable.
  Now we can do it thanks to GPU's. Thanks to the game industry developments cards have evolved to allow us to do the deep learning calculations. Not all GPU's are tailored for this. NVIDIA supports CUDA which are crafted for DL. In amazon p2 instances gives us that. 

All we need are examples (data) to make.

## Using AWS aliases 
**aws-get-p2:** Gets the AMI Id and saves it for future queries.

**aws-start:** Starts the instance based on the instance id from ^.

**aws-stop:** Stops ths AMI.

**aws-ip:** Gets the ip to navigate the AMI, and saves it for future usage.

**aws-ssh:** Connects to the server, using the ip from ^.

Note: On AWS we can use a t2.micro, to get started. It can be migrated later. 

--------
## Tackling a problem. 

- Split data into test and validation groups. 
- It's recommended to have a sample data. A small group to quickly test the models.
- If you use a pre-tranided model you'll be affected by the biases of their own data. 
------------
## Architecture of the stack
VGG16 - Keras - Theano - CUDA+cuDNN

Top<-----------------------------------|AMI Board

**VGG16:** Trained model to perform image detection of a known group/groups.

**Keras:** Parses Python and into code usable in by the backend in charge of the computations (thenos/tensrflow).
```javascript
// In the server: ~/.keras/keras.json
{
  backend: 'theano/tensorflow',
  image_dim_ordering: 'th/tf'
}
``` 
**Theano/Tensorflow:** Libraries especialized on running numerical computations for ML. Tensorflow is tailored for multi GPU implementations. 
```javascript
// In the server: ~/.theanorc

"device=gpu/cpu" // Depends on the AMI eg: a t2 instance doesn't have gpu.
```  

**CUDA:** Nvidias GPU implementation that anables fast computation of DL math. 

------------

>In data science a batch is a group of elements that are being analized. We devide our data set in batches to use all the power on a GPU. e.g: we can process 4 images at the same time. 


## Running in the server. 
- Connect to the AMI. 
- Start jupyter notebook. Ps: `dl_course`
- Copy the notebook. 
- Copy the data.

### Solved issues to get up and running
- After installing anaconda, you need to update your $PATH to make it run.
- `conda install nb_conda` to link conda with Jupyter. 

### Interesting resources 
image-net.org

