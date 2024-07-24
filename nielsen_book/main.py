from src.network import Network
from src.mnist_loader import load_data_wrapper


if __name__ == '__main__':
    print("Loading the MNIST dataset...")
    training_data, validation_data, test_data = load_data_wrapper()

    '''
    Creates a network with 3 layers:
        * 784 neurons in the first layer (28 x 28 pixel bitmap)
        * 30 neurons in the second (not sure why)
        * 10 neurons in the third (one for each digit)
    '''

    print("Initializing the neural network...")
    net = Network([784, 30, 10])
    net.stochastic_gradient_descent(
        training_data, 
        epochs=30, 
        mini_batch_size=10, 
        eta=3.0,  # learning rate: 3 is a good number, but figured out via trial and error
        test_data=test_data
    )