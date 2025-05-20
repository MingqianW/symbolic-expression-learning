Range of x1, x2 = (1, 2)
y = x1*x2:
Test loss is below machine precision, stopping training.
Learned symbolic expression:
1.018 * (x1^0.994 * x2^0.994) + (-0.010*x1 + -0.010\*x2 + 0.003)

y = x1*x2^(-1)
Epoch 17000, Loss: 0.000056 Test Loss: 0.000069
Test loss is below machine precision, stopping training.
Learned symbolic expression:
1.056 * (x1^0.983 * x2^-0.959) + (-0.032*x1 + 0.014\*x2 + -0.040)

y= x1^(1)_x2^(0.5)
Epoch 15400, Loss: 0.000000 Test Loss: 0.000000
Test loss is below machine precision, stopping training.
Learned symbolic expression:
1.053 _ (x1^1.001 * x2^0.478) + (-0.055*x1 + 0.001\*x2 + -0.000)

y = 0.8* x1^(0.3) * x2^(-0.6)
Epoch 24300, Loss: 0.000091 Test Loss: 0.000102
Test loss is below machine precision, stopping training.
Learned symbolic expression:
0.919 _ (x1^0.277 _ x2^-0.540) + (-0.011*x1 + 0.014*x2 + -0.122)

y = 0.8* x1^(0.3) * x2^(-0.6) + 0.5 _ x1 + 0.2 _ x2 + 0.1
Test loss is below machine precision, stopping training.
Learned symbolic expression:
0.919 _ (x1^0.277 _ x2^-0.540) + (0.489*x1 + 0.214*x2 + -0.022)

1. How we get the symbolic expression from the net. Mannully do? or use sympy?
2. For example if the true function is y = 0.8* x1^(0.3) * x2^(-0.6), we expect the power_activation to
   be the only "active" neuron, we expect the linear_out to have weight 0 and bias 0, and for the fully connect layer, we expect it has weight 1 and bias 0. We need to find a way to "encourage" this.
3. With the fc layer, the output struture is
   $\alpha_1 \cdot \left( C \cdot \prod_i x_i^{w_i} \right) + \alpha_2 \cdot \left( \sum_i a_i x_i + b \right) + \beta$,
   which is esentially $(\alpha_1 \cdot C) \cdot \prod_i x_i^{w_i} + \sum_i (\alpha_2 \cdot a_i) x_i + (\alpha_2 \cdot b + \beta)$.
   Instead of the fc layer, if we simply add power_out and linear_out, it would be
   $y = C \cdot \prod_i x_i^{w_i} + \sum_i a_i x_i + b$. They are almost same if we re-represent symbols.

True function: y = 0.8* x1^(0.3) * x2^(-0.6) + 0.5 _ x1 + 0.2 _ x2 + 0.1
With fc：
Full learned symbolic expression:
-0.833 * (x1^1.078 \_ x2^0.143) + 1.624*x1 + 0.038*x2 + 0.740
Without fc:
Test loss is below machine precision, stopping training.
Learned symbolic expression:
0.827 * (x1^0.312 \_ x2^-0.574) + (0.487*x1 + 0.201*x2 + 0.084)

True function: y = X _ R\*\*(-1) _ C\*_(-1). We let x1 = X, x2 = R, x3 = C
Without fc:
Test loss is below machine precision, stopping training.
Learned symbolic expression:
1.028 _ (x1^0.985 _ x2^-0.980 _ x3^-0.980) + (-0.010*x1 + 0.008*x2 + 0.008*x3 + -0.033)
With fc:
Epoch 10200, Loss: 0.000000 Test Loss: 0.000000
Test loss is below machine precision, stopping training.
Full learned symbolic expression:
0.380 * (x1^0.985 _ x2^-0.980 _ x3^-0.980) + -0.003*x1 + 0.002*x2 + 0.002\*x3 + -0.570

how to skip connection？

Lets say we have some linear transform on inputs of the true functions

True function: y = 0.8 _ (0.3 _ x1 + 1.4 _ x2 + 0.7 _ x3 + 0.6)\*_(0.5)
mixed model(original one):
Epoch 11200, Loss: 0.000005 Test Loss: 0.000005
Learned symbolic expression:
0.745 _ (x1^-0.014 _ x2^0.153 _ x3^-0.001) + (0.066*x1 + 0.192*x2 + 0.138\*x3 + 0.257)

as you can see, the result from the original model can't learn the expression
So we need to make the network deeper!

sprased pruned model:
Simplified symbolic expression:
y = 0.892 * ((0.210*x1 + 0.981*x2 + 0.490*x3 + 0.610))^0.533 (i.e., y3)

True function: y = 0.8 _ (0.3 _ x1 + 1.4 _ x2 + 0.7 _ x3 + 0.6)\*_(0.5)
mixed model(original one):
Epoch 11200, Loss: 0.000005 Test Loss: 0.000005
Learned symbolic expression:
0.745 _ (x1^-0.014 _ x2^0.153 _ x3^-0.001) + (0.066*x1 + 0.192*x2 + 0.138\*x3 + 0.257)

as you can see, the result from the original model can't learn the expression
So we need to make the network deeper!
Add Choice layer via the STE(https://arxiv.org/abs/1308.3432)
sprased pruned model:
Simplified symbolic expression:
y = 0.892 * ((0.210*x1 + 0.981*x2 + 0.490*x3 + 0.610))^0.533 (i.e., y3)

To print the sprase model: y = C2 _ (C1 _ PROD(xi^wi))^w2 + a2 * (SUM(ai*xi) + b1) + b2
and
y = C2 _ C1^w2 _ PROD(xi^{wi _ w2}) + a2 _ (SUM(ai*xi) + b1) + b2
= [C2 * C1^w2] _ PROD(xi^{wi _ w2}) + a2 * SUM(ai*xi) + a2 \* b1 + b2

Note we restrict all intermediate value to be positive, maybe there are better options: F.softplus?

Let's try if the network can distinguish different structure:
True function: y = 0.4 _ ( (x1\*\*0.3) _ (x2 ** 0.7) \* (x3 ** 0.6)) + 0.3
sprased pruned model:
Simplified symbolic expression:
y = 0.386 _ (x1^0.306 _ x2^0.715 \* x3^0.613) + 0.316 (i.e., y4)

Also, it can identify the irrelevant input variable:
True function: y = 0.4 _(x2 \*\* 0.7) _ (x3 \*_ 0.6) + 0.3
Simplified symbolic expression:
y = 0.384 _ (x1^0.000 _ x2^0.719 _ x3^0.616) + 0.318 (i.e., y4)
Simplified symbolic expression:
y = 0.843 * ((0.229*x1 + -0.000*x2 + 0.534*x3 + 0.631))^0.554 (i.e., y3)

When we have three layers, lets say its performance:
True function: y = 0.7 _ (0.4 _ ( (x1**0.3) \* (x2 ** 0.7) _ (x3 ** 0.6)) + 0.3) ** (0.5)
Final output:
switch_h selects: y5
y = 0.872 _ (0.263 _ 0.640 _ (x1^0.309 _ x2^0.725 _ x3^0.620) + 0.470)^0.847

It is basically two task,1. figure out the scture of function and how each variable contribute to each sub-part of function 2. calculate and get the correct coefficient of the expression

if the true coeffcient is not inside the range of parameter initilze, it is hard to learn
The initilization of paramter is super important. If "distances" from the paramter to its true value is different among paramters, it is hard to learn. In our example, it is even harder since different linear transformations can still easily achieve same value

1. Cutoff matters how to get a good cut off, we need to think about a good way to decide cutoff.
   a fixed threshold for a few epchos
   after swtiches unchanged for a certain number of epochs?
   The issue is the switch may get wrong even since eraly layer.
   Or even repeat the trail for maybe 13 times and majority vote?

2. We still need skip connection!
   feed the input varibles into later layers resulting an increasing network?

$$
$$
