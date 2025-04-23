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
3.  
   With the fc layer, the output struture is 
    $\alpha_1 \cdot \left( C \cdot \prod_i x_i^{w_i} \right) + \alpha_2 \cdot \left( \sum_i a_i x_i + b \right) + \beta$, 
    which is esentially $(\alpha_1 \cdot C) \cdot \prod_i x_i^{w_i} + \sum_i (\alpha_2 \cdot a_i) x_i + (\alpha_2 \cdot b + \beta)$. 
    Instead of the fc layer, if we simply add power_out and linear_out, it would be
   $y = C \cdot \prod_i x_i^{w_i} + \sum_i a_i x_i + b$. They are almost same if we re-represent symbols.

True function: y = 0.8* x1^(0.3) * x2^(-0.6) + 0.5 _ x1 + 0.2 _ x2 + 0.1
With fc：
Full learned symbolic expression:
-0.833 * (x1^1.078 _ x2^0.143) + 1.624*x1 + 0.038*x2 + 0.740
Without fc:
Test loss is below machine precision, stopping training.
Learned symbolic expression:
0.827 * (x1^0.312 _ x2^-0.574) + (0.487*x1 + 0.201*x2 + 0.084)


True function: y = X * R**(-1) * C**(-1). We let x1 = X, x2 = R, x3 = C 
Without fc: 
Test loss is below machine precision, stopping training.
Learned symbolic expression:
1.028 * (x1^0.985 * x2^-0.980 * x3^-0.980) + (-0.010*x1 + 0.008*x2 + 0.008*x3 + -0.033)
With fc:
Epoch 10200, Loss: 0.000000  Test Loss: 0.000000
Test loss is below machine precision, stopping training.
Full learned symbolic expression:
0.380 * (x1^0.985 * x2^-0.980 * x3^-0.980) + -0.003*x1 + 0.002*x2 + 0.002*x3 + -0.570

how to skip connection