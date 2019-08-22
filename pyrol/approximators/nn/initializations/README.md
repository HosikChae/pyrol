#### Initialization Schemes for Deep Neural Network Architectures
Notes:  
-Batch normalization makes the network much less sensitive to quality of initialization  
-Good initializations should have close to mean 0 and standard deviation 1  
-Add normalizaton layers is a good idea
-`Weights = Weights/ (Var) ** 0.5` should push standard deviation of output to 1 (see LSUV)
-By looking at the standard deviation equation can see how that's true (see LSUV)
-For the bias you can just subtract out the mean so output mean closer to 0 (see LSUV)