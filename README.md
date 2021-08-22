# Tensorflow-gpu-ge1080
Steps followed to install TensorFlow GPU on Windows 10 using Nvidia GeForce GTX 1080 card, Tensorflow 2.6, CUDA 11.2 and cuDNN 8.1. In this case, TensorFlow-GPU was installed for implementing deep learning models. There were many failures from the start till the end with several re-installations and restart of the system. 
1. First step is to figure out which combination of versions to be installed for TensorFlow, python, CUDA, visual studio and cuDNN. 
If visual studio, CUDA, TensorFlow and cuDNN versions are not matched, then we get dll error when we execute import TensorFlow. In the later stages, after successfully importing TensorFlow, errors may arise due to the mismatch of python, numpy and TensorFlow versions.

The below tested build configurations shows the matching versions of TensorFlow, python, Microsoft visual studio (MSVC), cuDNN and CUDA.
source: https://www.tensorflow.org/install/source_windows

Version	                Python version  Compiler	               Build tools	    cuDNN CUDA
tensorflow_gpu-2.6.0	3.6-3.9	MSVC 2019	                Bazel 3.7.2	        8.1	11.2
tensorflow_gpu-2.5.0	3.6-3.9	MSVC 2019	                Bazel 3.7.2	        8.1	11.2
tensorflow_gpu-2.4.0	3.6-3.8	MSVC 2019	                Bazel 3.1.0	        8.0	11.0
tensorflow_gpu-2.3.0	3.5-3.8	MSVC 2019	                Bazel 3.1.0	        7.6	10.1
tensorflow_gpu-2.2.0	3.5-3.8	MSVC 2019	                Bazel 2.0.0	        7.6	10.1
tensorflow_gpu-2.1.0	3.5-3.7	MSVC 2019	                Bazel 0.27.1-0.29.1	7.6	10.1
tensorflow_gpu-2.0.0	3.5-3.7	MSVC 2017	                Bazel 0.26.1	        7.4	10
tensorflow_gpu-1.15.0	3.5-3.7	MSVC 2017	                Bazel 0.26.1	        7.4	10
tensorflow_gpu-1.14.0	3.5-3.7	MSVC 2017	                Bazel 0.24.1-0.25.2	7.4	10
tensorflow_gpu-1.13.0	3.5-3.7	MSVC 2015 update 3	Bazel 0.19.0-0.21.0	7.4	10
tensorflow_gpu-1.12.0	3.5-3.6	MSVC 2015 update 3	Bazel 0.15.0	        7.2	9.0
tensorflow_gpu-1.11.0	3.5-3.6	MSVC 2015 update 3	Bazel 0.15.0	         7	9
tensorflow_gpu-1.10.0	3.5-3.6	MSVC 2015 update 3	Cmake v3.6.3	         7	9
tensorflow_gpu-1.9.0	3.5-3.6	MSVC 2015 update 3	Cmake v3.6.3	         7	9

I first installed visual studio 2017, cuda 10, cuDNN 7.4, TensorFlow 2.0. After successful installation of TensorFlow GPU and passing the checks related to GPU and cuda availability, I got import error on importing Keras requiring TensorFlow 2.2 or higher. 

Then I installed visual studio 2019, CUDA 11.2, cuDNN 8.1, tensorflow_gpu-2.5.0. 

Again after detecting GPU, successfully importing TensorFlow and Keras, I got this error while fitting LSTM model : "NotImplementedError: Cannot convert a symbolic Tensor (bidirectional/forward_lstm/strided_slice:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported". 
This was solved by reinstalling numpy and TensorFlow which got upgraded to 2.6.0 (source: tensorflow/models#9706).

In the official TensorFlow website, software requirements for CUDA® 11.2 requires Nvidia GPU driver version to be 450.80.02 or higher. The latest driver version installed for GTX 1080 card will be 471.68.  So we can install CUDA 11.2 version (I didn't try the latest CUDA version i.e., 11.4,  as many sources advised to stay away from the latest versions to avoid errors from bugs and fixes). 

2. To begin fresh installations, delete all the older versions of visual studio, Nvidia related drivers (cuda, cuDNN, gpu drivers). 
see https://www.youtube.com/watch?v=KZFn0dvPZUQ

a) Download Visual studio 2019 before installing Nvidia GPU drivers, CUDA and cuDNN -> -> https://visualstudio.microsoft.com/downloads/  
Click on community version for free download. Click install, continue without workloads.
After installation, restart the system.

b) Install Nvidia GPU driver -> https://www.nvidia.com/en-gb/geforce/drivers/
Enter GPU card information and click search. Download and install (471.68-desktop-win10-win11-64bit-international-nsd-dch-whql .exe) 
Follow on-screen prompts. Install graphics driver.. COMPLETE

c) Next Install CUDA 11.2.2  -> https://developer.nvidia.com/cuda-toolkit-archive
Select Windows, 10, exe (local) and download.
https://user-images.githubusercontent.com/87984816/130338919-a3018ede-fa00-46af-b069-32c6153e5661.PNG

d) Install cuDNN 8.1 by registering to Nvidia developer program membership here-> https://developer.nvidia.com/cuda-toolkit-archive
I downloaded cuDNN v8.1.1 (February 26th 2021), for CUDA 11.0, 11.1 and 11.2
cudnn-11.2-windows-x64-v8.1.1.33.zip

Inside the zip folder, there is a CUDA folder containing bin, include and lib folders. Copy and paste the files inside bin, lib and include of cuDNN to bin, lib and include of CUDA at C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2 respectively.

Then set path variables- For more details, look at these videos
 https://www.youtube.com/watch?v=qrkEYf-YDyI
https://www.youtube.com/watch?v=KZFn0dvPZUQ

In the official TensorFlow site, https://www.tensorflow.org/install/gpu
Add the CUDA®, CUPTI, and cuDNN installation directories to the %PATH% environmental variable. For example, if the CUDA® Toolkit is installed to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0 and cuDNN to C:\tools\cuda, update your %PATH% to match:

SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\extras\CUPTI\lib64;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include;%PATH%
SET PATH=C:\tools\cuda\bin;%PATH%

3. Install Anaconda latest version

Open anaconda command prompt
Create new environment on Python 3.8 (i didn't try 3.9, try if it works without any error)
conda create --name tf_gpu tensorflow-gpu==2.5.0
