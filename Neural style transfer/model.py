import tensorflow as tf
import utility as util

def nn_model(sess, input_image, model, objective, loss, num_iterations=200):
    
    J, J_content, J_style = loss

    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model['input'].assign(input_image))

    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        sess.run(objective)

        
        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])


        # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            util.save_image("output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    util.save_image('output/generated_image.jpg', generated_image)
    
    return generated_image