import torch
import torch.nn as nn
from inspect import signature
from network import MixedModel
import torch.optim as optim
from inspect import signature
from torch.optim.lr_scheduler import ReduceLROnPlateau


DOMAIN = (1, 2)    # Domain of dataset - range from which we sample x
# DOMAIN = np.array([[0, -1, -1], [1, 1, 1]])  # Use this format if each input variable has a different domain

def generate_data(func, N, range_min=DOMAIN[0], range_max=DOMAIN[1]):
    """Generates datasets."""
    x_dim = len(signature(func).parameters)     # Number of inputs to the function, or, dimensionality of x
    x = (range_max - range_min) * torch.rand([N, x_dim]) + range_min
    y = torch.tensor([[func(*x_i)] for x_i in x])
    return x, y #x.shape: [N, x_dim] and y.shape: [N, 1]


def true_func(x1, x2):
    return 0.8* x1**(0.3) * x2**(-0.6) + 0.5 * x1 + 0.2 * x2 + 0.1

def print_symbolic_expression(model):
    w = model.power_activation.w.detach().cpu().numpy()  # exponents
    C = model.power_activation.C.item()                  # scalar multiplier
    a = model.linear.weight.detach().cpu().numpy().flatten()  # linear weights
    b = model.linear.bias.item()                         # bias term

    input_terms = [f"x{i+1}" for i in range(len(w))]

    # Power term: C * x1^w1 * x2^w2 * ...
    power_expr = " * ".join([f"{x}^{w_i:.3f}" for x, w_i in zip(input_terms, w)])
    power_expr = f"{C:.3f} * ({power_expr})"

    # Linear term: a1*x1 + a2*x2 + ...
    linear_expr = " + ".join([f"{a_i:.3f}*{x}" for a_i, x in zip(a, input_terms)])
    linear_expr = f"({linear_expr} + {b:.3f})"

    # Full expression
    full_expr = f"{power_expr} + {linear_expr}"
    print("Learned symbolic expression:")
    print(full_expr)
    
# ---- Train the Model ----
def train_model():
    torch.manual_seed(233)
    
    x_train, y_train = generate_data(true_func, N=1000)
    input_dim = x_train.shape[1]
    model = MixedModel(input_dim)
    x_test, y_test = generate_data(true_func, N=200)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2) 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2000, verbose=True)

    
    num_epochs = 50000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(x_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(x_test)
            test_loss = criterion(y_pred, y_test)
            if test_loss < torch.finfo(torch.float32).eps:
                print("Test loss is below machine precision, stopping training.")
                break
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}  Test Loss: {test_loss.item():.6f}")
        scheduler.step(test_loss)

    print_symbolic_expression(model)
    return model

if __name__ == "__main__":
    model = train_model()
