import torch
import torch.nn as nn

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer (n)
            linear_1_out_features: the out features of first linear layer (d)
            linear_2_in_features: the in features of second linear layer (d)
            linear_2_out_features: the out features of second linear layer (k)
            f_function: string for the f function: relu | sigmoid | identity 
            g_function: string for the g function: relu | sigmoid | identity
        """

        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features), # [d * n]
            b1 = torch.randn(linear_1_out_features),      # [d * 1]
            W2 = torch.randn(linear_2_out_features, linear_2_in_features), # [k * d]
            b2 = torch.randn(linear_2_out_features),      # [k * 1]
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features), # [d * n]  T
            dJdb1 = torch.zeros(linear_1_out_features), # [d * 1]  T
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features), # [k * d] T
            dJdb2 = torch.zeros(linear_2_out_features), # [k * 1]  T
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        
        self.cache['batch_size'] = x.shape[0]
        self.cache['x'] = x
        self.cache['n'] = self.parameters['W1'].shape[1] 
        self.cache['d'] = self.parameters['W1'].shape[0] 
        self.cache['k'] = self.parameters['W2'].shape[0] 
        self.cache['all_z2'] = []  # list of tensors
        self.cache['dz2dz1'] = []
        self.cache['dy_hatdz3'] = []
        y_hat = torch.zeros(self.cache['batch_size'], self.cache['k'])  # [batch_size * k]
        
        for i in range(self.cache['batch_size']):
            self.cache['z1'] = torch.mv(self.parameters['W1'], x[i]) + self.parameters['b1']
            
            dz2dz1 = torch.zeros(self.cache['d'])
            dy_hatdz3 = torch.zeros(self.cache['k'])
            
            if self.f_function == 'relu':
                m = nn.ReLU()
                self.cache['z2'] = m(self.cache['z1'])
                for j in range(self.cache['d']):
                    if self.cache['z1'][j] >= 0:
                        dz2dz1[j] = 1

            elif self.f_function == 'sigmoid':  
                m = nn.Sigmoid()
                self.cache['z2'] = m(self.cache['z1'])
                dz2dz1 = torch.exp(-self.cache['z1'])*(1+torch.exp(-self.cache['z1']))**(-2)
            
            else:
                self.cache['z2'] = self.cache['z1']
                dz2dz1 = torch.ones(self.cache['d'])
            
            self.cache['all_z2'].append(torch.clone(self.cache['z2']))
            # [batch_size * tensor(d)]
            
            self.cache['dz2dz1'].append(dz2dz1)
                
            self.cache['z3'] = torch.mv(self.parameters['W2'], self.cache['z2']) + self.parameters['b2']
            
            if self.g_function == 'relu':
                m = nn.ReLU()
                y_hat[i] = m(self.cache['z3'])
                for j in range(self.cache['k']):
                    if self.cache['z3'][j] >= 0:
                        dy_hatdz3[j] = 1 
            elif self.g_function == 'sigmoid':
                m = nn.Sigmoid()
                y_hat[i] = m(self.cache['z3'])
                dy_hatdz3 = torch.exp(-self.cache['z3']) * (1 + torch.exp(-self.cache['z3']))**(-2) #[d]
            else:
                y_hat[i] = self.cache['z3']
                dy_hatdz3 = torch.ones(self.cache['k'])
            
            self.cache['dy_hatdz3'].append(dy_hatdz3)
                
        return y_hat
                                                            
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        for i in range(self.cache['batch_size']):
            dJdW1 = torch.zeros(self.cache['d'], self.cache['n']) # [d * n]  T
            dJdb1 = torch.zeros(self.cache['d']) # [d * 1]  T
            dJdW2 = torch.zeros(self.cache['k'], self.cache['d']) # [k * d] T
            dJdb2 = torch.zeros(self.cache['k']) # [k * 1]  T
            
            for j in range(self.cache['k']):
                dJdb2[j] = dJdy_hat[i][j].item() * self.cache['dy_hatdz3'][i][j].item()
                # tensor([a, b, c])     [k * 1]  torch.Size([k])
            
            dJdW2 = torch.mm(torch.unsqueeze(dJdb2, 1), torch.unsqueeze(self.cache['all_z2'][i], 1).t()) 
            tmp = torch.mm(torch.unsqueeze(dJdb2, 1).t(), self.parameters['W2']).t()     #[1* k, k * d].t()   [d * 1]
            
            for j in range(self.cache['d']):
                dJdb1[j] = tmp[j].item() * self.cache['dz2dz1'][i][j].item()     # [d * 1]
            
            dJdW1 = torch.mm(torch.unsqueeze(dJdb1, 1), torch.unsqueeze(self.cache['x'][i], 1).t()) 
            
            self.grads['dJdW1'] += dJdW1
            self.grads['dJdb1'] += dJdb1
            self.grads['dJdW2'] += dJdW2
            self.grads['dJdb2'] += dJdb2
        
        self.grads['dJdW1'] = self.grads['dJdW1']*(1/self.cache['batch_size'])
        self.grads['dJdb1'] = self.grads['dJdb1']*(1/self.cache['batch_size'])
        self.grads['dJdW2'] = self.grads['dJdW2']*(1/self.cache['batch_size'])
        self.grads['dJdb2'] = self.grads['dJdb2']*(1/self.cache['batch_size'])


    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    dJdy_hat = torch.mul(y_hat - y, 2)/y.shape[1]
    loss = torch.mean((y_hat - y)**2)
    
    return loss, dJdy_hat


def bce_loss(y, y_hat):
    """
    Args:
        y: the label tensor
        y_hat: the prediction tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    sum = torch.sum(y*torch.log(y_hat)+(1-y)*torch.log(1-y_hat))*(-1/y.shape[1])
    dJdy_hat = ((1-y)/(1-y_hat)-y/y_hat)/y.shape[1]
    
    loss = sum / y.shape[0]
    
    return loss, dJdy_hat

