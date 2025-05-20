import torch
import torch.nn as nn
import numpy as np
from inspect import signature
from network import MixedModel, SparseComposedModel, SparseComposedModelPruned
import torch.optim as optim
from inspect import signature
from torch.optim.lr_scheduler import ReduceLROnPlateau


DOMAIN = (1, 2)    # Domain of dataset - range from which we sample x
#DOMAIN = np.array([[1, 11, 47], [2, 14, 50]])  # Use this format if all input variables have the same domain
# DOMAIN = np.array([[0, -1, -1], [1, 1, 1]])  # Use this format if each input variable has a different domain

def generate_data(func, N, range_min=DOMAIN[0], range_max=DOMAIN[1]):
    """Generates datasets."""
    x_dim = len(signature(func).parameters)  # Number of inputs to the function, or, dimensionality of x
    range_min = torch.tensor(range_min, dtype=torch.float32)  # Convert to PyTorch tensor
    range_max = torch.tensor(range_max, dtype=torch.float32)  # Convert to PyTorch tensor
    x = (range_max - range_min) * torch.rand([N, x_dim]) + range_min
    y = torch.tensor([[func(*x_i)] for x_i in x])
    
    return x, y  # x.shape: [N, x_dim] and y.shape: [N, 1]


def true_func(x1,x2,x3):
    return (0.3 * x1 + 0.7 * x2 + 0.6 * x3 + 0.4)

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

def print_symbolic_expression_sparse(model):
    # ----- First layer -----
    w = model.power_activation.w.detach().cpu().numpy()
    C1 = model.power_activation.C.item()
    a = model.linear.weight.detach().cpu().numpy().flatten()
    b1 = model.linear.bias.item()
    input_terms = [f"x{i+1}" for i in range(len(w))]

    power_expr = " * ".join([f"{x}^{w_i:.3f}" for x, w_i in zip(input_terms, w)])
    power_expr = f"{C1:.3f} * ({power_expr})"

    linear_expr = " + ".join([f"{a_i:.3f}*{x}" for a_i, x in zip(a, input_terms)])
    linear_expr = f"({linear_expr} + {b1:.3f})"

    # ----- Switches -----
    c_selected = model.switch_c.selected()
    d_selected = model.switch_d.selected()
    e_selected = model.switch_e.selected()
    c_choice = ["power_out", "linear_out"][c_selected]
    d_choice = ["power_out", "linear_out"][d_selected]
    e_choice = ["y3", "y4"][e_selected]

    # ----- Second layer -----
    w2 = model.power_activation2.w.detach().cpu().numpy()[0]
    C2 = model.power_activation2.C.item()
    a2 = model.linear2.weight.detach().cpu().numpy().flatten()[0]
    b2 = model.linear2.bias.item()

    # Compose y3
    if c_selected == 0:
        # y3 = C2 * (C1 * PROD(xi^wi))^w2 = C2 * C1^w2 * PROD(xi^{wi*w2})
        y3_expr = f"{C2:.3f} * ({C1:.3f})^{w2:.3f} * (" + \
                  " * ".join([f"{x}^{w_i * w2:.3f}" for x, w_i in zip(input_terms, w)]) + ")"
        simplified_y3 = f"{C2 * (C1 ** w2):.3f} * (" + \
                        " * ".join([f"{x}^{w_i * w2:.3f}" for x, w_i in zip(input_terms, w)]) + ")"
    else:
        y3_expr = f"{C2:.3f} * ({linear_expr})^{w2:.3f}"
        simplified_y3 = y3_expr

    # Compose y4
    if d_selected == 0:
        y4_expr = f"{a2:.3f}*{power_expr} + {b2:.3f}"
        simplified_y4 = f"{a2 * C1:.3f} * (" + \
                        " * ".join([f"{x}^{w_i:.3f}" for x, w_i in zip(input_terms, w)]) + f") + {b2:.3f}"
    else:
        y4_expr = f"{a2:.3f}*{linear_expr} + {b2:.3f}"
        simplified_y4 = y4_expr

    # Final output (based on switch_e)
    if e_selected == 0:
        selected_expr = f"{simplified_y3}"
        selected_label = "y3"
    else:
        selected_expr = f"{simplified_y4}"
        selected_label = "y4"

    # ----- Output -----
    print("Learned symbolic expression (SparseComposedModel):")
    print("First layer:")
    print(f"  power_out = {power_expr}")
    print(f"  linear_out = {linear_expr}")
    print("Switches:")
    print(f"  Input to power_activation2: {c_choice}")
    print(f"  Input to linear2: {d_choice}")
    print(f"  Final output selected: {e_choice}")
    print("Second layer:")
    print(f"  y3 = {y3_expr}")
    print(f"  y4 = {y4_expr}")
    print("Simplified symbolic expression:")
    print(f"  y = {selected_expr}  (i.e., {selected_label})")


# def print_symbolic_expression_with_fc(model):
#     w = model.power_activation.w.detach().cpu().numpy()
#     C = model.power_activation.C.item()
#     a = model.linear.weight.detach().cpu().numpy().flatten()
#     b = model.linear.bias.item()

#     alpha = model.fc.weight.detach().cpu().numpy().flatten()
#     beta = model.fc.bias.item()

#     input_terms = [f"x{i+1}" for i in range(len(w))]

#     # Flattened contributions
#     final_C = alpha[0] * C
#     final_bias = alpha[1] * b + beta
#     final_a = alpha[1] * a

#     # Expression building
#     power_expr = " * ".join([f"{x}^{w_i:.3f}" for x, w_i in zip(input_terms, w)])
#     linear_expr = " + ".join([f"{a_i:.3f}*{x}" for a_i, x in zip(final_a, input_terms)])

#     full_expr = f"{final_C:.3f} * ({power_expr}) + {linear_expr} + {final_bias:.3f}"

#     print("Full learned symbolic expression:")
#     print(full_expr)

def print_symbolic_expression_pruned(model):
    w = model.power_activation.w.detach().cpu().numpy()
    C1 = model.power_activation.C.item()
    a = model.linear.weight.detach().cpu().numpy().flatten()
    b1 = model.linear.bias.item()

    input_terms = [f"x{i+1}" for i in range(len(w))]

    # First layer outputs
    power_expr = " * ".join([f"{x}^{w_i:.3f}" for x, w_i in zip(input_terms, w)])
    power_expr = f"{C1:.3f} * ({power_expr})"

    linear_expr = " + ".join([f"{a_i:.3f}*{x}" for a_i, x in zip(a, input_terms)])
    linear_expr = f"({linear_expr} + {b1:.3f})"

    # Based on the pruned switch indices
    c_selected = model.idx_c
    d_selected = model.idx_d
    e_selected = model.idx_e

    # Second layer compositions
    w2 = model.power_activation2.w.detach().cpu().numpy()[0]
    C2 = model.power_activation2.C.item()
    a2 = model.linear2.weight.detach().cpu().numpy().flatten()[0]
    b2 = model.linear2.bias.item()

    # Compose y3
    if c_selected == 0:
        # input_to_power2 = power_out
        y3_expr = f"{C2:.3f} * ({C1:.3f})^{w2:.3f} * (" + \
                  " * ".join([f"{x}^{w_i * w2:.3f}" for x, w_i in zip(input_terms, w)]) + ")"
        simplified_y3 = f"{C2 * (C1 ** w2):.3f} * (" + \
                        " * ".join([f"{x}^{w_i * w2:.3f}" for x, w_i in zip(input_terms, w)]) + ")"
    else:
        # input_to_power2 = linear_out
        y3_expr = f"{C2:.3f} * ({linear_expr})^{w2:.3f}"
        simplified_y3 = y3_expr

    # Compose y4
    if d_selected == 0:
        # input_to_linear2 = power_out
        y4_expr = f"{a2:.3f}*{power_expr} + {b2:.3f}"
        simplified_y4 = f"{a2 * C1:.3f} * (" + \
                        " * ".join([f"{x}^{w_i:.3f}" for x, w_i in zip(input_terms, w)]) + f") + {b2:.3f}"
    else:
        # input_to_linear2 = linear_out
        y4_expr = f"{a2:.3f}*{linear_expr} + {b2:.3f}"
        simplified_y4 = y4_expr

    # Final output
    if e_selected == 0:
        selected_expr = simplified_y3
        selected_label = "y3"
    else:
        selected_expr = simplified_y4
        selected_label = "y4"

    # ----- Print nicely -----
    print("Learned symbolic expression (SparseComposedModelPruned):")
    print("First layer:")
    print(f"  power_out = {power_expr}")
    print(f"  linear_out = {linear_expr}")
    print("Second layer:")
    print(f"  y3 = {y3_expr}")
    print(f"  y4 = {y4_expr}")
    print("Simplified symbolic expression:")
    print(f"  y = {selected_expr}  (i.e., {selected_label})")

def train_model(model_type="mixed"):
    torch.manual_seed(233)
    
    x_train, y_train = generate_data(true_func, N=200000)
    input_dim = x_train.shape[1]
    if model_type == "mixed":
        model = MixedModel(input_dim)
    elif model_type == "sparse":
        model = SparseComposedModel(input_dim)
    else:
        raise ValueError("Unknown model_type: choose 'mixed' or 'sparse'")
    x_test, y_test = generate_data(true_func, N=5000)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2000)

    cutoff = 0.001   # Cutoff for pruning need to be chose carefully or maybe should design a good way to decide
    pruned = False
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
        if model_type == "sparse" and not pruned and test_loss.item() < cutoff:
            print(f"Pruning model at epoch {epoch}, test loss = {test_loss.item():.6f}")
            model = SparseComposedModelPruned(model)
            optimizer = optim.Adam(model.parameters(), lr=1e-2)  # Reinitialize optimizer
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8000)   #Reinitialize scheduler
            pruned = True
        # Switches can get stuck early. add some noise to the logits during early training:
        if epoch < 5000 and model_type == "sparse" and not pruned:
            with torch.no_grad():
                noise = torch.randn_like(model.switch_c.logits) * 0.5
                model.switch_c.logits.add_(noise)
                noise = torch.randn_like(model.switch_d.logits) * 0.5
                model.switch_d.logits.add_(noise)
                noise = torch.randn_like(model.switch_e.logits) * 0.5
                model.switch_e.logits.add_(noise)
        if epoch % 100 == 0 and model_type == "mixed":
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}  Test Loss: {test_loss.item():.6f}")
            print_symbolic_expression(model)
        if epoch % 100 == 0 and model_type == "sparse" and not pruned:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}  Test Loss: {test_loss.item():.6f}")
            print_symbolic_expression_sparse(model)
        if epoch % 100 == 0 and pruned and model_type == "sparse":
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}  Test Loss: {test_loss.item():.6f}")
            print_symbolic_expression_pruned(model)
        scheduler.step(loss)

    if model_type == "mixed":
        print_symbolic_expression(model)
    elif model_type == "sparse" and pruned:
        print_symbolic_expression_pruned(model)
    else:
        print_symbolic_expression_sparse(model)
    return model

if __name__ == "__main__":
    model_type = "sparse"  # or "mixed"
    num_trials = 5
    all_expressions = []

    for i in range(num_trials):
        print(f"\n========== Trial {i+1} ==========")
        seed = 233 + i  # Change seed each time
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = train_model(model_type=model_type)

        # Capture expression as string
        import io
        import contextlib

        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            if model_type == "mixed":
                print_symbolic_expression(model)
            elif isinstance(model, SparseComposedModelPruned):
                print_symbolic_expression_pruned(model)
            else:
                print_symbolic_expression_sparse(model)
            expression_str = buf.getvalue()

        all_expressions.append(expression_str)

    # Optionally print all expressions at the end
    print("\n========== Summary of All Learned Expressions ==========")
    for i, expr in enumerate(all_expressions, 1):
        print(f"\n--- Trial {i} ---")
        print(expr)
