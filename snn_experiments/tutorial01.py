'''
This code is taken / adapted from the SNNTorch tutorial
https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_1_spikegen.ipynb
'''

import snntorch as snn
from snntorch import utils
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


'''
1. Training Parameters
'''
batch_size=128
data_path='data/mnist'
num_classes = 10  # MNIST has 10 output classes

'''
2. Torch Variables
'''
dtype = torch.float


'''
3. Define a transform
'''
transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

''' 
4. Download dataset
'''
mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
print(mnist_train)


'''
5. SNNTorch has some convenience functions that allow us to subset the training data.
Apply data_subset to reduce the dataset by the factor defined in subset. 
E.g., for subset=10, a training set of 60,000 will be reduced to 6,000.
'''
subset = 100
mnist_train = utils.data_subset(mnist_train, subset)
print(f"The size of mnist_train is {len(mnist_train)}")


'''
6. Create a dataloader, which will serve up the data in batches. 
DataLoaders in PyTorch are a handy interface for passing data into a network. 
They return an iterator divided up into mini-batches of size batch_size.
'''
# recall that batch_size is 128 (set above):
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)


'''
7. Spike Encoding: Rate Encoding
OK, now the data (pixel map) needs to be comverted into spike encoding, where each pixel
gets encoded into a series of spikes. We're going to use the "Rate Coding"
method where input intensity is converted into a firing rate or spike count.
Let's keep it simple:

Let's assume that each pixel is represented by 4 possible spikes:

Black Pixel (value 0.0)     Gray Pixel (value 0.5)      White Pixel (value 1.0)
1  0  0  0                  0  1  0  1                  1  1  1  1
__ __ __ __                 __ __ __ __                 __ __ __ __
t1 t2 t3 t4                 t1 t2 t3 t4                 t1 t2 t3 t4
'''

# Temporal Dynamics (assume that each pixel is represented by a maximum of 10 spikes)
num_steps = 10
# create vector filled with 0.5
raw_vector = torch.ones(num_steps)*0.5
print(raw_vector)

# pass each sample through a Bernoulli trial
# expected value is 5 spikes (5/10) but could be more or less...
rate_coded_vector = torch.bernoulli(raw_vector)
print(f"Converted vector: {rate_coded_vector}")
print(f"The output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time.")

# if we increase the number of steps (as n -> infinity), 
# the value approaches 0.5 more often:
num_steps = 100
raw_vector = torch.ones(num_steps)*0.5
rate_coded_vector = torch.bernoulli(raw_vector)
print(f"The output is spiking {rate_coded_vector.sum()*100/len(rate_coded_vector):.2f}% of the time.")


'''
8. Another way of doing rate-encoded spikes using the built-in SNN function...
'''
def output_bitmap(pixel_maps_it, labels_it, num_digits=10, delimiter=""):
    cell_width = 2 + len(delimiter)
    for i in range(num_digits):
        bitmap = pixel_maps_it[i][0]
        w = 28*(cell_width + 1)
        
        print('-'*w, end='\n')
        print("Labeled as:", int(labels_it[i]))
        print('-'*w, end='\n')
        for row in bitmap:
            for cell in row:
                if cell == 0:
                    print(' '*cell_width, end=delimiter)
                else:
                    print(f'{cell:2.1f}', end=delimiter)
            print()
            if len(delimiter) > 0:
                print('-'*w, end='\n')

def output_spike_videos(spike_data, num_digits=10):
    # print(spike_data[0])

    # let's take a look at what a "spike version of an image might look like..."
    import matplotlib.pyplot as plt
    import snntorch.spikeplot as splt

    for i in range(num_digits):
        spike_data_sample = spike_data[:, i, 0]
        print(spike_data_sample.size())

        torch.Size([100, 28, 28])

        fig, ax = plt.subplots()
        anim = splt.animator(spike_data_sample, fig, ax, interval=40)
        movie_path = f'snn_experiments/figures/rate_encoded_demo_{i}.m4a'
        anim.save(movie_path)
        print('open', movie_path, 'in QuickTime to view anumation')
    #help(anim)

from snntorch import spikegen

# Iterate through minibatches (recall, each minibatch has 128 images)
data = iter(train_loader)
pixel_maps_it, labels_it = next(data)

# print the first 5 digits as bitmaps
output_bitmap(pixel_maps_it, labels_it, num_digits=5, delimiter="|")

# Convert to spiking data using snntorch's spikegen function:
spike_data = spikegen.rate(pixel_maps_it, num_steps=num_steps)

# make spike videos:
print('generating videos...')
output_spike_videos(spike_data, num_digits=5,)



'''
9. Another way to encode information that is much more energy / memory efficient is to use
latency-encoded spikes. So, instead of having each image represented as 100 1s and 0s, you just encode 
the spike once: big pixel value means early spike; small input means late spike.

1.0: |________
0.5: ____|____
0.0: ________|

'''