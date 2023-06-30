import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import cmath
import math
import time

from scipy import stats
from scipy.signal import convolve2d
from PIL import Image

n=288
pi=math.pi
PSF=np.zeros([n,n])
centre=n/2+1
radius=57
random.seed(1)

#initialize 
for i in range(288):
    for j in range(288):
        dist= ((i-centre)**2+(j-centre)**2)**(1/2)
        M=stats.norm.pdf(round(dist), 0, 1/6*n)
        if dist<=radius:
            phi=random.uniform(-1,1)*pi/2
            c=complex(0,2*pi*phi)
            PSF[i,j]=M*cmath.exp(c)

psf=np.fft.fft2(PSF)
psf=np.abs(psf)
psf=np.square(psf)
normalizator=np.sum(psf)
psf_norm=psf*normalizator

plt.imshow(psf_norm, interpolation='nearest', cmap='gray')
plt.show()
        

#print(di)
#i=Image.fromarray(di)
#i.show()

df_train=pd.read_csv('train.csv')
pixel_data=df_train.iloc[:,1:785] #from col 1 to 785
label=df_train["label"]
#There are 42000 picture inside



def convolution(image, i, psf_norm=psf_norm):
    output=np.zeros_like('image')
    output = convolve2d(image,psf_norm, mode='same', boundary='symm')
    normalizator2=np.sum(output)
    output=output*normalizator2
    input_dir='img_500_2'
    output_dir = 'img_500_2_conv'
    filename = str(label[i])+"_"+str(i)+".png"
    plt.imsave(os.path.join(output_dir, filename), output, cmap='gray')
    plt.imsave(os.path.join(input_dir, filename), image, cmap='gray')

start_time = time.time()

for i in range(500, 1000):
    image=pixel_data.iloc[i].to_numpy()
    image=np.reshape(image,(28,28))
    convolution(image, i, psf_norm=psf_norm)
    
end_time = time.time()

elapsed_time = end_time - start_time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))
#last progress 999



