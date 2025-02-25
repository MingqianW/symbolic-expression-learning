1. Does $f(\bar{x}) = \bar{f(x)}$? Does this depends on the distribution of x?
   Ok, its clearly false counter example: $\frac{3^{2} + 2^{2}}{2} = 6.5 \neq 2.5^{2} $ Why? Jensen's inequality!
2. Normalization: do we need to normaliza X,R, and C?

CNN is used to process spatial data (such as images), but the data here is one-dimensional, so a one-dimensional convolutional layer should be used.

However, considering that there are very few input features (only 3), using too deep convolutional layers may not be appropriate, and fully connected layers may be more effective. So maybe only 1-2 convolutional layers are needed, followed by fully connected layers.

Simple CNN is just a simple 1D CNN where has nothing spefical. It has similar structure with the usual image classfication CNN. This should have bad performance. Since I did not utilize the characteristic of the current data.
Yes, it is very bad:Epoch 30 | Loss: 1110626304.0000 | Val Loss: 3301698299.8710


Next lets try to consturct a CNN with proper design. One thing to notice here, do we need to consider the "Time serise"
property of data. That is to say, X_t = X_t-1 + f(X_t-1, R_t-1, C_t-1) and X_t+1 = X_t + f(X_t, R_t, C_t). X_t is both "input" and "output". For now, we totally ingore this property and when we prapare the data, we shuffle the data for training(See data_loader.py). If we need to consider that property, RNN or even LSTM.??? Idk, need more research on it! skip it for now.

![image info](CNNstructureguess.jpg)


Next, consider the picture above, it is natural to extract feature for each X, R, C separately. Then the question is how can we use X,R,C nodes to get y. First, conv layer will not work here since it has only 3 nodes. I would say KAN are probably the better method, but lets try direct connect first.


#Todo: 1. finish  simple CNN: done

2. make sure dataset is designed as I expected: done

3. design CNN: 

4. the requirement file
