import torch
import torch.nn as nn
import numpy as np
from inspect import signature
from network import SparseComposedModel3Layer, SparseComposedModel3LayerPruned
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

DOMAIN = (1, 2)

def generate_data(func, N, range_min=DOMAIN[0], range_max=DOMAIN[1]):
    x_dim = len(signature(func).parameters)
    range_min = torch.tensor(range_min, dtype=torch.float32)
    range_max = torch.tensor(range_max, dtype=torch.float32)
    x = (range_max - range_min) * torch.rand([N, x_dim]) + range_min
    y = torch.tensor([[func(*x_i)] for x_i in x])
    return x, y


def true_func(x1, x2, x3):
    return 0.7 * (0.4 * ( (x1**0.3) * (x2 ** 0.7)  * (x3 ** 0.6)) + 0.3) ** (0.5)


def print_symbolic_expression_pruned_3layer(model):
    w1 = model.power_activation1.w.detach().cpu().numpy()
    C1 = model.power_activation1.C.item()
    a1 = model.linear1.weight.detach().cpu().numpy().flatten()
    b1 = model.linear1.bias.item()
    input_terms = [f"x{i+1}" for i in range(len(w1))]

    power_expr1 = " * ".join([f"{x}^{w:.3f}" for x, w in zip(input_terms, w1)])
    linear_expr1 = " + ".join([f"{a:.3f}*{x}" for a, x in zip(a1, input_terms)])

    # switches
    c, d, f, g, h = model.idx_c, model.idx_d, model.idx_f, model.idx_g, model.idx_h

    # second layer
    w2 = model.power_activation2.w.detach().cpu().numpy()[0]
    C2 = model.power_activation2.C.item()
    a2 = model.linear2.weight.detach().cpu().numpy().flatten()[0]
    b2 = model.linear2.bias.item()

    y3_expr = (
        f"{C2:.3f} * ({C1:.3f})^{w2:.3f} * (" + 
        " * ".join([f"{x}^{w1[i] * w2:.3f}" for i, x in enumerate(input_terms)]) + ")"
        if c == 0 else
        f"{C2:.3f} * ({linear_expr1} + {b1:.3f})^{w2:.3f}"
    )
    y4_expr = (
        f"{a2:.3f} * {C1:.3f} * (" + 
        " * ".join([f"{x}^{w:.3f}" for x, w in zip(input_terms, w1)]) + f") + {b2:.3f}"
        if d == 0 else
        f"{a2:.3f} * ({linear_expr1} + {b1:.3f}) + {b2:.3f}"
    )

    # third layer
    w3 = model.power_activation3.w.detach().cpu().numpy()[0]
    C3 = model.power_activation3.C.item()
    a3 = model.linear3.weight.detach().cpu().numpy().flatten()[0]
    b3 = model.linear3.bias.item()

    y5_expr = (
        f"{C3:.3f} * ({y3_expr})^{w3:.3f}"
        if f == 0 else
        f"{C3:.3f} * ({y4_expr})^{w3:.3f}"
    )
    y6_expr = (
        f"{a3:.3f} * ({y3_expr}) + {b3:.3f}"
        if g == 0 else
        f"{a3:.3f} * ({y4_expr}) + {b3:.3f}"
    )

    final_expr = y5_expr if h == 0 else y6_expr

    print("Learned symbolic expression (SparseComposedModel3LayerPruned):")
    print("Layer 1:")
    print(f"  power_out1 = {C1:.3f} * ({power_expr1})")
    print(f"  linear_out1 = ({linear_expr1} + {b1:.3f})")
    print("Layer 2:")
    print(f"  y3 = {y3_expr}")
    print(f"  y4 = {y4_expr}")
    print("Layer 3:")
    print(f"  y5 = {y5_expr}")
    print(f"  y6 = {y6_expr}")
    print("Final output:")
    print(f"  y = {final_expr}")


def print_symbolic_expression_sparse_3layer(model):
    w1 = model.power_activation1.w.detach().cpu().numpy()
    C1 = model.power_activation1.C.item()
    a1 = model.linear1.weight.detach().cpu().numpy().flatten()
    b1 = model.linear1.bias.item()
    input_terms = [f"x{i+1}" for i in range(len(w1))]

    power_expr1 = " * ".join([f"{x}^{w:.3f}" for x, w in zip(input_terms, w1)])
    linear_expr1 = " + ".join([f"{a:.3f}*{x}" for a, x in zip(a1, input_terms)])

    # switches
    c, d, f, g, h = [s.selected() for s in [
        model.switch_c, model.switch_d, model.switch_f, model.switch_g, model.switch_h
    ]]

    # second layer
    w2 = model.power_activation2.w.detach().cpu().numpy()[0]
    C2 = model.power_activation2.C.item()
    a2 = model.linear2.weight.detach().cpu().numpy().flatten()[0]
    b2 = model.linear2.bias.item()

    y3_expr = (
        f"{C2:.3f} * ({C1:.3f})^{w2:.3f} * (" + 
        " * ".join([f"{x}^{w1[i] * w2:.3f}" for i, x in enumerate(input_terms)]) + ")"
        if c == 0 else
        f"{C2:.3f} * ({linear_expr1} + {b1:.3f})^{w2:.3f}"
    )
    y4_expr = (
        f"{a2:.3f} * {C1:.3f} * (" + 
        " * ".join([f"{x}^{w:.3f}" for x, w in zip(input_terms, w1)]) + f") + {b2:.3f}"
        if d == 0 else
        f"{a2:.3f} * ({linear_expr1} + {b1:.3f}) + {b2:.3f}"
    )

    # third layer
    w3 = model.power_activation3.w.detach().cpu().numpy()[0]
    C3 = model.power_activation3.C.item()
    a3 = model.linear3.weight.detach().cpu().numpy().flatten()[0]
    b3 = model.linear3.bias.item()

    y5_expr = (
        f"{C3:.3f} * ({y3_expr})^{w3:.3f}"
        if f == 0 else
        f"{C3:.3f} * ({y4_expr})^{w3:.3f}"
    )
    y6_expr = (
        f"{a3:.3f} * ({y3_expr}) + {b3:.3f}"
        if g == 0 else
        f"{a3:.3f} * ({y4_expr}) + {b3:.3f}"
    )

    final_expr = y5_expr if h == 0 else y6_expr

    print("Learned symbolic expression (SparseComposedModel3Layer):")
    print("Layer 1:")
    print(f"  power_out1 = {C1:.3f} * ({power_expr1})")
    print(f"  linear_out1 = ({linear_expr1} + {b1:.3f})")
    print("Layer 2:")
    print(f"  y3 = {y3_expr}")
    print(f"  y4 = {y4_expr}")
    print("Switches:")
    print(f"  switch_c: {'power_out1' if c == 0 else 'linear_out1'}")
    print(f"  switch_d: {'power_out1' if d == 0 else 'linear_out1'}")
    print("Layer 3:")
    print(f"  y5 = {y5_expr}")
    print(f"  y6 = {y6_expr}")
    print("Switches:")
    print(f"  switch_f: {'y3' if f == 0 else 'y4'}")
    print(f"  switch_g: {'y3' if g == 0 else 'y4'}")
    print("Final output:")
    print(f"  switch_h selects: {'y5' if h == 0 else 'y6'}")
    print(f"  y = {final_expr}")


def train_model():
    torch.manual_seed(2200)
    
    x_train, y_train = generate_data(true_func, N=200000)
    input_dim = x_train.shape[1]
    model = SparseComposedModel3Layer(input_dim)
    x_test, y_test = generate_data(true_func, N=5000)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2000, verbose=True)
    
    
    cutoff = 0.000001   # Cutoff for pruning need to be chose carefully or maybe should design a good way to decide
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
            if torch.isnan(test_loss):
                print("Test loss is NaN, stopping training.")
                break
        if  not pruned and test_loss.item() < cutoff:
            print(f"Pruning model at epoch {epoch}, test loss = {test_loss.item():.6f}")
            model = SparseComposedModel3LayerPruned(model)
            optimizer = optim.Adam(model.parameters(), lr=1e-2)  # Reinitialize optimizer
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8000, verbose=True)   #Reinitialize scheduler
            pruned = True
        # Switches can get stuck early. add some noise to the logits during early training:
        if epoch < 5000 and not pruned:
            with torch.no_grad():
                noise = torch.randn_like(model.switch_c.logits) * 0.5
                model.switch_c.logits.add_(noise)
                noise = torch.randn_like(model.switch_d.logits) * 0.5
                model.switch_d.logits.add_(noise)
                noise = torch.randn_like(model.switch_f.logits) * 0.5
                model.switch_f.logits.add_(noise)
                noise = torch.randn_like(model.switch_g.logits) * 0.5
                model.switch_g.logits.add_(noise)
        if epoch % 100 == 0 and not pruned:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}  Test Loss: {test_loss.item():.6f}")
            print_symbolic_expression_sparse_3layer(model)
        if epoch % 100 == 0 and pruned :
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}  Test Loss: {test_loss.item():.6f}")
            print_symbolic_expression_pruned_3layer(model)
        scheduler.step(loss)

    if pruned:
        print_symbolic_expression_pruned_3layer(model)
    else:
        print_symbolic_expression_sparse_3layer(model)
    return model


if __name__ == "__main__":
    num_trials = 5
    all_expressions = []

    for i in range(num_trials):
        print(f"\n========== Trial {i+1} ==========")
        seed = 2200 + i  # Change seed each time
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = train_model()

        # Capture symbolic expression
        import io
        import contextlib

        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            if isinstance(model, SparseComposedModel3LayerPruned):
                print_symbolic_expression_pruned_3layer(model)  # Update to support 3 layers if not already
            else:
                print_symbolic_expression_sparse_3layer(model)  # Update to support 3 layers if not already
            expression_str = buf.getvalue()

        all_expressions.append(expression_str)

    print("\n========== Summary of All Learned Expressions ==========")
    for i, expr in enumerate(all_expressions, 1):
        print(f"\n--- Trial {i} ---")
        print(expr)
