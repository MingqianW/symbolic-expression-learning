import torch
import torch.nn as nn
import numpy as np
from inspect import signature
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Specified_EQL.network import SpecifiedSctureModel1, SpecifiedSctureModel2 
# run this via command line: python -m Specified_EQL.train
    
DOMAIN = (1, 2)

def generate_data(func, N, range_min=DOMAIN[0], range_max=DOMAIN[1]):
    x_dim = len(signature(func).parameters)
    range_min = torch.tensor(range_min, dtype=torch.float32)
    range_max = torch.tensor(range_max, dtype=torch.float32)
    x = (range_max - range_min) * torch.rand([N, x_dim]) + range_min
    y = torch.tensor([[func(*x_i)] for x_i in x])
    return x, y

def true_func(x1, x2, x3):
    # Term 1: product of sums raised to powers
    t1a = (0.2 * x1 + 0.5 * x2 + 0.3 * x3) ** 0.6
    t1b = (0.6 * x1 + 0.1 * x2 + 0.8 * x3) ** 0.4
    term1 = 1.1 * (t1a * t1b)

    # Term 2: linear term
    term2 = 0.3 * x1 + 0.7 * x2 + 0.2 * x3

    # Term 3: direct power product
    term3 =  (x1 ** 0.5) * (x2 ** 1.2) * (x3 ** 0.3)

    return term1 + term2 + term3

def print_symbolic_expression(model):
    import torch.nn as nn
    import numpy as np

    input_terms = [f"x{i+1}" for i in range(model.input_dim)]

    # Term 1: alpha * prod(sum a_ij x_j)^w_i
    A = model.linear_sum.weight.detach().cpu().numpy()
    w = model.power_sum.w.detach().cpu().numpy()
    term1_factors = []
    for i in range(model.input_dim):
        inner_sum = " + ".join([f"{A[i,j]:.3f}*{input_terms[j]}" for j in range(model.input_dim)])
        term1_factors.append(f"({inner_sum})^{w[i]:.3f}")
    term1_expr = " * ".join(term1_factors)

    # Term 3: c * prod x_i^{y_i}
    y = model.power_direct.w.detach().cpu().numpy()
    term3_factors = [f"{input_terms[i]}^{y[i]:.3f}" for i in range(model.input_dim)]
    term3_expr = " * ".join(term3_factors)

    if hasattr(model, "final_linear"):  # SpecifiedSctureModel2
        final_w = model.final_linear.weight.detach().cpu().numpy().flatten()
        alpha = final_w[0]
        c = final_w[1]
        beta = final_w[2:]

        term1_full = f"{alpha:.3f} * {term1_expr}"
        term2_expr = " + ".join([f"{beta[i]:.3f}*{input_terms[i]}" for i in range(model.input_dim)])
        term3_full = f"{c:.3f} * {term3_expr}"

    else:  # SpecifiedSctureModel1
        alpha = model.power_sum.C.item()
        c = model.power_direct.C.item()
        beta = model.linear_term.weight.detach().cpu().numpy().flatten()

        term1_full = f"{alpha:.3f} * {term1_expr}"
        term2_expr = " + ".join([f"{beta[i]:.3f}*{input_terms[i]}" for i in range(model.input_dim)])
        term3_full = f"{c:.3f} * {term3_expr}"

    full_expr = f"{term1_full} + ({term2_expr}) + {term3_full}"
    print("Symbolic expression:")
    print(full_expr)


def train_model(model_class, num_epochs=20000):
    torch.manual_seed(233)
    x_train, y_train = generate_data(true_func, N=200000)
    x_test, y_test = generate_data(true_func, N=5000)

    input_dim = x_train.shape[1]
    model = model_class(input_dim)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2000)

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
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}")
                print_symbolic_expression(model)

        scheduler.step(loss)

    print("\nFinal learned expression:")
    print_symbolic_expression(model)
    return model

if __name__ == "__main__":
    print("Training SpecifiedSctureModel1:")
    train_model(SpecifiedSctureModel1)

    print("\nTraining SpecifiedSctureModel2:")
    train_model(SpecifiedSctureModel2)
