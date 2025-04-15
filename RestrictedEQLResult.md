Range of x1, x2 = (1, 2)
y = x1*x2:
Test loss is below machine precision, stopping training.
Learned symbolic expression:
1.018 * (x1^0.994 * x2^0.994) + (-0.010*x1 + -0.010*x2 + 0.003)

y = x1*x2^(-1)
Epoch 17000, Loss: 0.000056  Test Loss: 0.000069
Test loss is below machine precision, stopping training.
Learned symbolic expression:
1.056 * (x1^0.983 * x2^-0.959) + (-0.032*x1 + 0.014*x2 + -0.040)

y= x1^(1)*x2^(0.5)
Epoch 15400, Loss: 0.000000  Test Loss: 0.000000
Test loss is below machine precision, stopping training.
Learned symbolic expression:
1.053 * (x1^1.001 * x2^0.478) + (-0.055*x1 + 0.001*x2 + -0.000)

y = 0.8* x1^(0.3) * x2^(-0.6)
Epoch 24300, Loss: 0.000091  Test Loss: 0.000102
Test loss is below machine precision, stopping training.
Learned symbolic expression:
0.919 * (x1^0.277 * x2^-0.540) + (-0.011*x1 + 0.014*x2 + -0.122)

y = 0.8* x1^(0.3) * x2^(-0.6) + 0.5 * x1 + 0.2 * x2 + 0.1
Test loss is below machine precision, stopping training.
Learned symbolic expression:
0.919 * (x1^0.277 * x2^-0.540) + (0.489*x1 + 0.214*x2 + -0.022)