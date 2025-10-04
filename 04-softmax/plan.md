softmax optimization in C++ CUDA

I want to create a andrej karapathy style project that I could finish in a day. It should be focused at optimization of softmax in neural networks. My goal is to learn cuda along the way instead of just copy-pasting the code. Please consider following details while designing this project-
- I want to test it with a neural network preferrably CNN on large dataset like tiny-imagenet
- I want to have one standard version of the network with no optimizations.
- I want to use techniques like coalesced memory access, thread reduction and online softmax for calculation of max value out of all logits.
- I want to have comparison of computational time among all the modifications to softmax.
- I should be able to also try kernel fusion if it is possible to implement in any stage of the neural network.
- Later I should be able to extend this project to transformers.
- I want to host this project on github so I would like to have a readme which explains all the approaches taken explanation of the code snippets like - 
- You may use following referrences:
    - https://arxiv.org/pdf/1805.02867
    - https://openreview.net/forum?id=XsNA2b8GPz&referrer=%5Bthe+profile+of+Mert+Pilanci%5D%28%2Fprofile%3Fid%3D%7EMert_Pilanci1%29
    - https://github.com/Maharshi-Pandya/cudacodes
    - https://pytorch.org/blog/flashattention-3/
    - https://iiswc.org/iiswc2022/IISWC2022_63.pdf