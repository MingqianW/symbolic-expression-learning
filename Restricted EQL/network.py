import torch
import torch.nn as nn
import layers

class MixedModel(nn.Module):
    def __init__(self, input_dim):
        super(MixedModel, self).__init__()
        self.power_activation = layers.PowerActivation(input_dim)
        self.linear = nn.Linear(input_dim, 1)
        self.fc = nn.Linear(2, 1) 


    def forward(self, x):
        power_out = self.power_activation(x)  # [N, 1]
        linear_out = self.linear(x)           # [N, 1]
       #combined = torch.cat([power_out, linear_out], dim=1)  #  [N, 2]
        #return self.fc(combined)  # [N, 1]
        return power_out + linear_out  # Should I just add them? Yes!
    
class SparseComposedModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.power_activation = layers.PowerActivation(input_dim)
        self.linear = nn.Linear(input_dim, 1)
        with torch.no_grad():
            self.linear.weight.abs_() 
            self.linear.bias.abs_() 
        self.switch_c = layers.ChoiceLayerSTE()
        self.switch_d = layers.ChoiceLayerSTE()
        self.switch_e = layers.ChoiceLayerSTE()
        self.power_activation2 = layers.PowerActivation(1)
        self.linear2 = nn.Linear(1, 1) #Note the weight and bias are initilized in the range of [0,0.577)
        with torch.no_grad():
            self.linear2.weight.abs_() 
            self.linear2.bias.abs_() 
            
        # Set switch logits so switch_c and/or switch_d select linear_out at initialization.
        #why alyway   Input to power_activation2: power_out;Input to linear2: linear_out
        # I mean we should not encourage that same operation to be selected consecutively
                    
        self.switch_c.logits.data = torch.tensor([0.0, 3.0])
        self.switch_d.logits.data = torch.tensor([3.0, 0.0]) 
        self.switch_e.logits.data = torch.tensor([1.0, 1.0]) # for the final output, we dont encourage it to be linear


    def forward(self, x):
        power_out = self.power_activation(x)   # [N, 1]
        linear_out = self.linear(x)            # [N, 1]
        
        # Switches select which previous output goes to next stage
        input_to_power2 = self.switch_c(power_out, linear_out)
        input_to_linear2 = self.switch_d(power_out, linear_out)
        
        y3 = self.power_activation2(input_to_power2)
        y4 = self.linear2(input_to_linear2)
        
        return self.switch_e(y3, y4)  # [N, 1]
    

class SparseComposedModelPruned(nn.Module):
    def __init__(self, orig_model):
        super().__init__()
        self.power_activation = layers.PowerActivation(orig_model.input_dim)
        self.linear = nn.Linear(orig_model.input_dim, 1)
        
        # with torch.no_grad():   #Note the weight and bias are initilized in the range of [0,0.577)
        #     self.linear.weight.abs_()
        with torch.no_grad():
            nn.init.uniform_(self.linear.weight, a=0.0, b=1.0)
            if self.linear.bias is not None:
                nn.init.uniform_(self.linear.bias, a=0.0, b=1.0) 
        self.power_activation2 = layers.PowerActivation(1)
        self.linear2 = nn.Linear(1, 1)
        with torch.no_grad():
            nn.init.uniform_(self.linear2.weight, a=0.0, b=1.0)
            if self.linear.bias is not None:
                nn.init.uniform_(self.linear2.bias, a=0.0, b=1.0)

        # Store the fixed switch decisions
        self.idx_c = orig_model.switch_c.selected()
        self.idx_d = orig_model.switch_d.selected()
        self.idx_e = orig_model.switch_e.selected()

    def forward(self, x):
        power_out = self.power_activation(x)
        linear_out = self.linear(x)

        # Hard-wired based on selected switches
        input_to_power2 = power_out if self.idx_c == 0 else linear_out
        input_to_linear2 = power_out if self.idx_d == 0 else linear_out

        y3 = self.power_activation2(input_to_power2)
        y4 = self.linear2(input_to_linear2)

        # Final fixed output
        return y3 if self.idx_e == 0 else y4

class SparseComposedModel3Layer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        # ----- Layer 1 -----
        self.power_activation1 = layers.PowerActivation(input_dim)
        self.linear1 = nn.Linear(input_dim, 1)
        with torch.no_grad():
            self.linear1.weight.abs_()
            self.linear1.bias.abs_()

        # ----- Layer 2 -----
        self.switch_c = layers.ChoiceLayerSTE()
        self.switch_d = layers.ChoiceLayerSTE()
        self.power_activation2 = layers.PowerActivation(1)
        self.linear2 = nn.Linear(1, 1)
        with torch.no_grad():
            self.linear2.weight.abs_()
            self.linear2.bias.abs_()

        # ----- Layer 3 -----
        self.switch_f = layers.ChoiceLayerSTE()
        self.switch_g = layers.ChoiceLayerSTE()
        self.power_activation3 = layers.PowerActivation(1)
        self.linear3 = nn.Linear(1, 1)
        with torch.no_grad():
            self.linear3.weight.abs_()
            self.linear3.bias.abs_()

        # ----- Final switch (Layer 4) -----
        self.switch_h = layers.ChoiceLayerSTE()

        # Init switches
        self.switch_c.logits.data = torch.tensor([0.0, 3.0])
        self.switch_d.logits.data = torch.tensor([3.0, 0.0])
        self.switch_f.logits.data = torch.tensor([0.0, 3.0])
        self.switch_g.logits.data = torch.tensor([3.0, 0.0])
        self.switch_h.logits.data = torch.tensor([1.0, 1.0])

    def forward(self, x):
        # ----- Layer 1 -----
        power_out1 = self.power_activation1(x)
        linear_out1 = self.linear1(x)

        # ----- Layer 2 -----
        input_to_power2 = self.switch_c(power_out1, linear_out1)
        input_to_linear2 = self.switch_d(power_out1, linear_out1)
        y3 = self.power_activation2(input_to_power2)
        y4 = self.linear2(input_to_linear2)

        # ----- Layer 3 -----
        input_to_power3 = self.switch_f(y3, y4)
        input_to_linear3 = self.switch_g(y3, y4)
        y5 = self.power_activation3(input_to_power3)
        y6 = self.linear3(input_to_linear3)

        # ----- Final output -----
        return self.switch_h(y5, y6)

class SparseComposedModel3LayerPruned(nn.Module):
    def __init__(self, orig_model):
        super().__init__()
        self.input_dim = orig_model.input_dim

        # ----- Layer 1 -----
        self.power_activation1 = layers.PowerActivation(self.input_dim)
        self.linear1 = nn.Linear(self.input_dim, 1)
        with torch.no_grad():
            nn.init.uniform_(self.linear1.weight, a=0.0, b=1.0)
            if self.linear1.bias is not None:
                nn.init.uniform_(self.linear1.bias, a=0.0, b=1.0)

        # ----- Layer 2 -----
        self.power_activation2 = layers.PowerActivation(1)
        self.linear2 = nn.Linear(1, 1)
        with torch.no_grad():
            nn.init.uniform_(self.linear2.weight, a=0.0, b=1.0)
            if self.linear2.bias is not None:
                nn.init.uniform_(self.linear2.bias, a=0.0, b=1.0)

        # ----- Layer 3 -----
        self.power_activation3 = layers.PowerActivation(1)
        self.linear3 = nn.Linear(1, 1)
        with torch.no_grad():
            nn.init.uniform_(self.linear3.weight, a=0.0, b=1.0)
            if self.linear3.bias is not None:
                nn.init.uniform_(self.linear3.bias, a=0.0, b=1.0)

        # ----- Store switch decisions -----
        self.idx_c = orig_model.switch_c.selected()
        self.idx_d = orig_model.switch_d.selected()
        self.idx_f = orig_model.switch_f.selected()
        self.idx_g = orig_model.switch_g.selected()
        self.idx_h = orig_model.switch_h.selected()

    def forward(self, x):
        # ----- Layer 1 -----
        power_out1 = self.power_activation1(x)
        linear_out1 = self.linear1(x)

        # ----- Layer 2 -----
        input_to_power2 = power_out1 if self.idx_c == 0 else linear_out1
        input_to_linear2 = power_out1 if self.idx_d == 0 else linear_out1
        y3 = self.power_activation2(input_to_power2)
        y4 = self.linear2(input_to_linear2)

        # ----- Layer 3 -----
        input_to_power3 = y3 if self.idx_f == 0 else y4
        input_to_linear3 = y3 if self.idx_g == 0 else y4
        y5 = self.power_activation3(input_to_power3)
        y6 = self.linear3(input_to_linear3)

        # ----- Final output -----
        return y5 if self.idx_h == 0 else y6