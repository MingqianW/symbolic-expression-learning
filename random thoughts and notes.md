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
Epoch 30 | Loss: nan | Val Loss: nan 


why we get nan as loss？1. did not normalize. 2. architure.

#Todo: 1. finish  simple CNN: done

2. make sure dataset is designed as I expected: done

3. design CNN: done

4. the requirement file done


As what I talked with Prof Singh in last meeting, I will 
1. include the noise I into the training: Done
2. Instead of X_t+1 = X_t + f(X_t, R_t, C_t), use X_t+1 = f(X_t, R_t, C_t): Done

Next, as talked before, I will not use the two model we have before, I will then try to use the EQL.
Thanks to their pytorch implementation(https://github.com/samuelkim314/DeepSymRegTorch/blob/main/utils/symbolic_network.py), my work become much easier!

Note: I use different version of matplotlib and h5py, I hope it is ok.

When the set of primative function does not conatin the true expression, it failed.

To be more specfic, it failed since there is no division in the previous set, even though 1/x cant be found.
Epoch: 12000    Total training loss: 61.393044  Test error: 1048664.500000 
clearly, it failed to learn the pattern. maybe need to fine-tune?
Guess: numerical stability? maybe consider try to do in log space

need to change the original function to other form without divison to try. 
Need more experiments.

1. Make sure EQL_main is correct first, try X_t+1 = X_t + 1 +I 
While if I use my own data generator, I get Epoch: 0        Total training loss: nan        Test error: nan

I can't figure out why.
Since I failed to run EQL_main on our own data generator, I would use their data generator first(The main difference is that it is not a time series data and each x_t is independent)

OK use its original benchmark file lets first test on 1/x to make sure it can learn divison. After a few tuning, it may not be able to learn the devision.

Maybe we can try only keep the identity and division to see if the modle can learn: Failed

Maybe it is just one of the issue of EQL, I will try KAN next.

Back to EQL, if I only allow use activation sets with 3 layers:        self.activation_funcs = [
            *[functions.Constant()] * 2,
            *[functions.Identity()] * 4,
            *[functions.Inverse()] * 2,
        ]
we can briefly get a symbolic expression that implies the division structure: 3.78596/(9.31071*x + 1.0e-6) - 4.24765/(1.0e-6 - 7.16598*x)
transforming to log space failed.

If we scale the data in to the range of (1,2), it is doable.

product need two input？ how to solve？

When there are only 1 variables, it is good, but when we have more than 2, it gets nasty.
Rescale the data before applying to the activation function
