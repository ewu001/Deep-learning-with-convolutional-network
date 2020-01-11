import tensorflow as tf
import imageio

import utility as util
import nst_service as service
import model as md


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

NUM_ITERATION = 40
TOTAL_COST_ALPHA = 10
TOTAL_COST_BETA = 40

# Create an Interactive Session
# Load the content image
# Load the style image
# Randomly initialize the image to be generated
# Load the VGG16 model
# Build the TensorFlow graph:
# Run the content image through the VGG16 model and compute the content cost
# Run the style image through the VGG16 model and compute the style cost
# Compute the total cost
# Define the optimizer and the learning rate
# Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.


# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()

content_image = imageio.imread("images/louvre_small.jpg")
content_image = util.reshape_and_normalize_image(content_image)

style_image = imageio.imread("images/monet.jpg")
style_image = util.reshape_and_normalize_image(style_image)

generated_image = util.generate_noise_image(content_image)

model = util.load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
print(model)


# Select the output tensor of layer conv4_2
out = model['conv3_4']

# Assign the content image to be the input of the VGG model.  
sess.run(model['input'].assign(content_image))
# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# Compute the content cost
J_content = service.compute_content_cost(a_C, a_G)

# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))
# Compute the style cost
J_style = service.compute_style_cost(model, STYLE_LAYERS, sess)

J = service.total_cost(J_content, J_style, TOTAL_COST_ALPHA, TOTAL_COST_BETA)


# define optimizer and train_step
optimizer = tf.train.AdamOptimizer(2.0)
objective = optimizer.minimize(J)

md.nn_model(sess, generated_image, model, objective, [J, J_content, J_style], NUM_ITERATION)
