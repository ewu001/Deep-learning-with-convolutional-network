Playground for Residual Network following the architecture of ResNet50

Residual network runs on skip connections to pass previous layer's activation directly into later layer before its activation step as a residual with the purpose of fighting vanishing gradient in deep convolution networks. 

By doing do, gradients can be directly backpropogated to earlier layers. There're two main types of residual block

One is identity block, it is the standard block used in Residual networks, and corresponds to the case where the input activation has the same dimension as the output activation. 

The other is convolutional block, it is used when input activation has different dimension as the output activation. In this case, the residual input is required to go through another convolutional layer to resize the input before it is added to the later layer before its activation step. Batch normalization may still be used but usually it does not go through non-linear activation because the purpose is to apply a learnt linear function to reshape the input. 

In this project, after compile step, the defined ResNet is trained to 2 separate data sets, one is MNIST with 70,000 images, the other is hand digit sign data set with 10,80 images.

Deep residual network is introduced in paper Deep Residual Learning for Image Recognition, archived at https://arxiv.org/pdf/1512.03385.pdf
