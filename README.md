# DeepModels
This is the repository for creating neural networks, which are later used for DeepConcolic tool.
To run please install set up conda environment as specified in https://github.com/sielos/DeepConcolic. 

In order to generate DNN models for MNIST dataset use command:
```
python mnist_DNNs.py
```
In order to generate DNN models for cats vs dogs dataset use command:
```
python main.py
```
In order to generate DNN models for cifar10 dataset use command:
```
python cifar10_DNNs.py
```

The generated models will appear under the models directory. 

To use these models for DeepConcolic tool copy the model file (.h5) to DeepConcolic/saved_models/ directory.
