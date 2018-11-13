import os
import numpy as np
import pandas as pd
from skimage import io
from keras.layers import Input, Dense, Flatten, Dropout, Reshape, Concatenate, UpSampling2D
from keras.layers import BatchNormalization, Activation, Conv2D, Conv2DTranspose,MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def load_MNIST():
    path = r'C:\Users\Aydin\Desktop\Face Datasets\MNIST\minimal'
    fileNames = [f for f in sorted(os.listdir(path))]
    images = np.array([np.array(io.imread(path + '\\' + f)) for f in fileNames]) #, as_grey= True
    #reshape
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    labels = np.array([int(f[6]) for f in fileNames])
    return images, labels

def generate_noise(n_samples, noise_dim):
    return np.random.normal(0, 1, size=(n_samples, noise_dim))

def generate_random_labels(labels_num, n):
    z = np.zeros((n, labels_num))
    x = np.random.choice(labels_num, n)
    for i in range(n):    
        z[i, x[i]] = 1    
    return z

def get_generator(input_layer, condition_layer):
    merged_input = Concatenate()([input_layer, condition_layer])  

    hid = Dense(8*8*128, activation='relu')(merged_input)
    hid = Reshape((8,8,128))(hid)
    hid = Conv2DTranspose(128, kernel_size=(5,5), strides=2, activation=LeakyReLU(alpha=0.1))(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = Conv2DTranspose(128, kernel_size=(5,5), strides=1, activation=LeakyReLU(alpha=0.1))(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    hid = Conv2DTranspose(128, kernel_size=(4,4), strides=1, activation=LeakyReLU(alpha=0.1))(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    out = Conv2DTranspose(1, kernel_size=(3,3), strides=1, activation='tanh')(hid)
    
    model = Model(inputs=[input_layer, condition_layer], outputs=out)
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_generator2(input_layer, condition_layer):
    merged_input = Concatenate()([input_layer, condition_layer])  
    
    hid = Dense(128*7*7, input_dim=110, activation=LeakyReLU(0.2))(merged_input)
    hid = BatchNormalization()(hid)
    hid = Reshape((7,7,128))(hid)
    hid = UpSampling2D()(hid)
    hid = Conv2D(64, 5, 5, border_mode='same', activation=LeakyReLU(0.2))(hid)
    hid = BatchNormalization()(hid)
    hid = UpSampling2D()(hid)
    out = Conv2D(1, 5, 5, border_mode='same', activation='tanh')(hid)
    
    return Model(inputs=[input_layer, condition_layer], outputs=out)

def get_discriminator(input_layer, condition_layer):
    hid = Conv2D(128, (5, 5), activation=LeakyReLU(alpha=0.1))(input_layer)
    hid = BatchNormalization(momentum=0.9)(hid)
    
    hid = Conv2D(128, (5, 5), activation=LeakyReLU(alpha=0.1))(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    
    hid = Conv2D(128, (5, 5), activation=LeakyReLU(alpha=0.1))(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    
    hid = MaxPooling2D(pool_size=(2, 2))(hid)
    
    hid = Dropout(0.25)(hid)
    hid = Flatten()(hid)
    
    hid = Dense(128, activation='relu')(hid)
    hid = BatchNormalization(momentum=0.9)(hid)
    
    hid = Dropout(0.5)(hid)
    merged_layer = Concatenate()([hid, condition_layer])
    hid = Dense(256, activation='relu')(merged_layer)
    hid = BatchNormalization(momentum=0.9)(hid)
    #hid = Dropout(0.4)(hid)
    out = Dense(1, activation='sigmoid')(hid)
    model = Model(inputs=[input_layer, condition_layer], outputs=out)
  
    return model

def get_discriminator2(input_layer, condition_layer):
    hid = Conv2D(64, 5, 5, subsample=(2,2), input_shape=(28,28,1), border_mode='same', activation=LeakyReLU(0.2))(input_layer)
    hid = Dropout(0.3)(hid)
    hid = Conv2D(128, 5, 5, subsample=(2,2), border_mode='same', activation=LeakyReLU(0.2))(hid)
    hid = Dropout(0.3)(hid)
    hid = Flatten()(hid)
    merged_layer = Concatenate()([hid, condition_layer])
    hid = Dense(256, activation=LeakyReLU(alpha=0.1))(merged_layer)
    hid = BatchNormalization(momentum=0.9)(hid)
    #hid = Dropout(0.4)(hid)
    out = Dense(1, activation='sigmoid')(hid)
    model = Model(inputs=[input_layer, condition_layer], outputs=out)
  
    return model

def get_GAN(img_shape, condition_shape):
    
    G_noise_input = Input(shape=(100,))
    G_condition_input = Input(shape=condition_shape)
    G = get_generator2(G_noise_input, G_condition_input)
    
    D_img_input = Input(shape=img_shape)
    D_condition_input = Input(shape=condition_shape)
    
    D = get_discriminator2(D_img_input, D_condition_input)
    D.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    D.trainable = False
    
    GAN_input = Input(shape=(100,))
    temp = G([GAN_input, G_condition_input])
    GAN_out = D([temp, D_condition_input])
    GAN = Model(inputs=[GAN_input, G_condition_input, D_condition_input], outputs=GAN_out)
    GAN.compile(optimizer='adam', loss='binary_crossentropy')
    return GAN, G, D

###############################################################################
#                                     START                                   #
###############################################################################

num_classes = 10
img_shape = (28,28,1)

GAN, G, D = get_GAN(img_shape, (num_classes,))

# Load data
X_train, y_train = load_MNIST()

# Normalize data
X_train = (X_train- 127.5) / 127.5

# Encode labels
y_train = np.array(pd.get_dummies(y_train))
 
# Training constants
epochs = 100
num_batches = 50
batch_size = int(len(X_train) / num_batches)

D_real_loss = []
D_generated_loss = []
G_loss = []

for epoch in range(epochs):
    print('-'*30)
    print(f'Epoch: {epoch}')
    for batch_num in range(num_batches):
        print(f'\tBatch: {batch_num}')
        
        D.trainable = True
        
        # Train Discriminator on real images
        real_images = X_train[batch_num*batch_size : (batch_num+1)*batch_size]
        labels = y_train[batch_num*batch_size : (batch_num+1)*batch_size]
        D_real_loss.append(D.train_on_batch([real_images, labels], np.ones((batch_size, 1))))
        print(f'\t\tD_real_loss: {D_real_loss[-1]}')
        
        # Train Discriminator on generated images
        noise_data = generate_noise(batch_size, 100)
        random_labels = generate_random_labels(num_classes, batch_size)
        generated_images = G.predict([noise_data, labels])
        D_generated_loss.append(D.train_on_batch([generated_images, labels],  np.zeros((batch_size, 1))))
        print(f'\t\tD_generated_loss: {D_generated_loss[-1]}')
        
        D.trainable = False
        
        # Train Generator
        noise_data = generate_noise(batch_size, 100)
        random_labels = generate_random_labels(num_classes, batch_size)
        G_loss.append(GAN.train_on_batch([noise_data, random_labels, random_labels], np.ones((batch_size, 1))))
        print(f'\t\tG_loss: {G_loss[-1]}')
        
    labels = np.array(pd.get_dummies(range(num_classes)))
    noise = generate_noise(10,100)
    prediction = G.predict([noise, labels])
    fig = plt.figure()
    plt.suptitle(f'Epoch: {epoch}')
    for i in range(num_classes):
        plt.subplot(1,10,i+1)
        plt.title(f'{i}')
        plt.imshow(prediction[i,:,:,0], cmap=plt.cm.binary)
        plt.axis('off')
        plt.savefig(f'Generated\\Epoch_{epoch}')
    
        


#
#GAN.predict([noise, y_train[[0]], y_train[[0]]])
        
#test
#
#G_noise_input = Input(shape=(100,))
#G_condition_input = Input(shape=(10,))
#G = get_generator2(G_noise_input, G_condition_input)
#
#













