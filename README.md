# Install Tensorflow-GPU on Windows 10 with Nvidia GeForce GTX1080
The steps followed to install TensorFlow GPU on Windows 10 using Nvidia GeForce GTX 1080 card, Tensorflow 2.6, CUDA 11.2 and cuDNN 8.1 are detailed here. TensorFlow-GPU was installed for implementing deep learning models. It took several trials of software re-installations and system restart to get this working. 
1. The first step is to find out which combination of versions need to be installed for TensorFlow, python, CUDA, visual studio and cuDNN. If visual studio, CUDA, TensorFlow and cuDNN versions are not matched, we get dll error while importing TensorFlow. Some error might occur in the later stages after successfully importing TensorFlow due to the mismatch of Python, numpy and TensorFlow versions.

The matching versions of TensorFlow, python, Microsoft visual studio (MSVC), cuDNN and CUDA are shown below.
source: https://www.tensorflow.org/install/source_windows

![build config](https://user-images.githubusercontent.com/87984816/130339580-45f8c445-209c-40a6-80ef-fd19b714d547.PNG)

The official TensorFlow website https://www.tensorflow.org/install/gpu, mentions CUDA® 11.2 requires Nvidia GPU driver version to be 450.80.02 or higher. The latest driver version installed for GTX 1080 card was 471.68.  So we can install CUDA 11.2 version for GTX 1080 card(I didn't try the latest CUDA version i.e., 11.4,  as many sources advised to stay away from the latest versions to avoid errors from bugs and fixes). 

2. To begin fresh installations, delete all the older versions of visual studio, Nvidia related drivers (cuda, cuDNN, gpu drivers). 
For more information see https://www.youtube.com/watch?v=KZFn0dvPZUQ

3. Install Visual studio 2019 before installing Nvidia GPU drivers, CUDA and cuDNN. Link to download visual studio is here -> https://visualstudio.microsoft.com/downloads/  
Click on community version for free download.

![MS VISUAL STUDIO 2019 download](https://user-images.githubusercontent.com/87984816/130339708-5889321e-4909-4fb6-b7ce-a9ebb7596344.PNG)

 Click install
	
![MS VISUAL STUDIO 2019_step1](https://user-images.githubusercontent.com/87984816/130339710-17e68ec7-5753-4965-9804-d3a40663326a.PNG)

continue without workloads

![MS VISUAL STUDIO 2019_step2](https://user-images.githubusercontent.com/87984816/130339711-6b505e3e-1218-485e-a6b3-11514110d5bc.PNG)

After completing the Visual Studio installation, restart the system.

4. Install Nvidia GPU driver from here -> https://www.nvidia.com/en-gb/geforce/drivers/
Enter GPU card information and click search. Download and install (471.68-desktop-win10-win11-64bit-international-nsd-dch-whql .exe)

![nvidiagpu_driver install](https://user-images.githubusercontent.com/87984816/130339522-e30640f2-d8cb-4857-8b50-be433780f16f.PNG)

Follow on-screen prompts. Install graphics driver.. COMPLETE

5. Next Install CUDA 11.2.2 from here  -> https://developer.nvidia.com/cuda-toolkit-archive

Select Windows, 10, exe (local) and download.
![cuda install step1](https://user-images.githubusercontent.com/87984816/130339509-3c640730-355c-4937-8d67-5c43fec2c923.PNG)


6. Install cuDNN 8.1 by registering to Nvidia developer program membership here-> https://developer.nvidia.com/rdp/cudnn-archive

I downloaded cuDNN v8.1.1 (February 26th 2021), for CUDA 11.0, 11.1 and 11.2
cudnn-11.2-windows-x64-v8.1.1.33.zip

Create new folder C:\tools\  and unzip contents to here.

![cudNN install step1](https://user-images.githubusercontent.com/87984816/130339623-507c6f89-a4eb-42a1-8121-76b5ef0098ac.PNG)

Inside the zip folder, there is a CUDA folder containing bin, include and lib folders. Copy and paste the files inside bin, lib and include of cuDNN to bin, lib and include of CUDA at C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2 respectively.


7. Then set path variables:

Search -> type - Edit the system environment variables -> click Environment Variables -> Click Path under System Variables


![set path variable](https://user-images.githubusercontent.com/87984816/130339631-e435b9dc-0753-4613-81b6-31fde54f6ee2.PNG)


Click New and paste the following (edit 11.2 if using other version)

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\extras\CUPTI\lib64

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include

C:\tools\cuda\bin


![set path variable2](https://user-images.githubusercontent.com/87984816/130339639-f0815c32-78b2-4d2f-a26f-83d50b11e871.PNG)

For more details, look at these videos:

https://www.youtube.com/watch?v=qrkEYf-YDyI

https://www.youtube.com/watch?v=KZFn0dvPZUQ

Source: TensorFlow site, https://www.tensorflow.org/install/gpu

Check CUDA version using nvidia-smi and nvcc --version 

Here is the link for more information https://varhowto.com/check-cuda-version/

8. Install Anaconda latest version from here -> https://www.anaconda.com/products/individual

Download and install, click on "Add Anaconda3 to my path environment variable"

![Anaconda installer_step2](https://user-images.githubusercontent.com/87984816/130340542-9f86d9a1-2279-411e-98c8-48ba1b38bb8f.PNG)


Add path to environment variable

C:\  ..... \anaconda3

After installation, open anaconda command prompt.
Create new environment on Python 3.8 (i didn't try 3.9, try if it works without any error)

conda create --name tf_gpu python=3.8 
conda activate tf_gpu
conda install tensorflow-gpu==2.5.0

Validation

Type "python" in tf_gpu environment

import tensorflow as tf 

You should get this message:  "I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudart64_110.dll"
(if you get dll error, then you have not set the path variables right or not saved the cuDNN files in CUDA folders )
check if CUDA and GPU are present. 

print(tf.test.is_gpu_available())

print(tf.test.is_built_with_cuda())

- you should see 'True' for both.

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

Result:  
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: NVIDIA GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1

from tensorflow.python.client import device_lib 

device_lib.list_local_devices()

you should see something like this:

[name: "/device:CPU:0"
 device_type: "CPU"
 memory_limit: 268435456
 locality {
 }
 incarnation: 1659933096265405359,
 name: "/device:GPU:0"
 device_type: "GPU"
 memory_limit: 6930300928
 locality {
   bus_id: 1
   links {
   }
 }
 incarnation: 10992939894467130346
 physical_device_desc: "device: 0, name: NVIDIA GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1"]


Finally run your deep learning model "model.fit" and check GPU performance on Task Manager.

I got this error "NotImplementedError: Cannot convert a symbolic Tensor (bidirectional/forward_lstm/strided_slice:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported."

After re-installation using

pip uninstall tensorflow

pip install --upgrade tensorflow (source: https://www.tensorflow.org/install/pip#virtual-environment-install)

pip uninstall numpy

pip install numpy

I got Tensor Flow Version: 2.6.0, Keras Version: 2.6.0, Python 3.8.11, Numpy 1.21.2, GPU is available.
Solved the error and model (bidirectional LSTM) was fitted.

These links are helpful to solve errors at this stage. Restarting the system might help too.

https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc

https://github.com/tensorflow/models/issues/9706

Some more useful links:

https://medium.com/featurepreneur/install-tensorflow-with-gpu-support-for-deep-learning-on-windows-10-d1d443fd5878

https://github.com/kartikvega/TensorFlow-Install-on-Win-10-w-1080TI

https://discuss.tensorflow.org/t/tensorflow-2-5-with-gpu-device-python-3-9-cuda-11-2-2-cudnn-8-1-1-conda-environment-windows-10/1385


