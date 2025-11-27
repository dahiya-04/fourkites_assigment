import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LossLandscapeProber:
    """
    A class for efficiently probing the loss landscape of a PyTorch model.
    Focuses on the Hessian-Vector Product (HVP) and Power Iteration for
    estimating the maximum eigenvalue (sharpness).
    """
    def __init__(self, model, loss_fn, data_loader):
        self.model = model
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.device = next(model.parameters()).device

    def _get_full_loss(self):
        """Computes the full loss over the entire dataset."""
        total_loss = 0.0
        num_samples = 0
        for inputs, targets in self.data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)
        return total_loss / num_samples

       # --- robust flatten helper ---
    def _get_flat_params(self):
        """Returns a flattened tensor of all model parameters (contiguous)."""
        return torch.cat([p.contiguous().view(-1) for p in self.model.parameters()])

    # --- robust hessian_vector_product ---
    def hessian_vector_product(self, vector):
        """
        Computes the Hessian-Vector Product (H * v).
        `vector` should be a list of tensors matching model.parameters() shapes.
        This implementation flattens both grad and v to compute g^T v robustly.
        """
        self.model.zero_grad()

        try:
            inputs, targets = next(iter(self.data_loader))
        except StopIteration:
            raise ValueError("Data loader is empty.")
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)

        # 1) First-order gradients (create_graph=True because we need second derivatives)
        grad_params = torch.autograd.grad(loss, self.model.parameters(), create_graph=True, allow_unused=True)

        # flatten grad_params safely (replace None with zeros)
        grad_list = []
        for g, p in zip(grad_params, self.model.parameters()):
            if g is None:
                grad_list.append(torch.zeros_like(p).contiguous().view(-1))
            else:
                grad_list.append(g.contiguous().view(-1))
        grad_flat = torch.cat(grad_list)

        # 2) Flatten input 'vector' (list of tensors) to a single vector
        vector_list = []
        for v, p in zip(vector, self.model.parameters()):
            # ensure shapes match and that v is contiguous
            if v is None:
                vector_list.append(torch.zeros_like(p).contiguous().view(-1))
            else:
                vector_list.append(v.contiguous().view(-1))
        vector_flat = torch.cat(vector_list)

        # ensure same device and dtype
        vector_flat = vector_flat.to(grad_flat.device, dtype=grad_flat.dtype)

        # 3) compute scalar g^T v robustly
        grad_v_dot_product = torch.dot(grad_flat, vector_flat)

        # 4) second-order: gradient of (g^T v) w.r.t. parameters -> H * v
        hvp_params = torch.autograd.grad(grad_v_dot_product, self.model.parameters(), allow_unused=True)

        # flatten hvp_params safely (None -> zeros)
        hvp_list = []
        for h, p in zip(hvp_params, self.model.parameters()):
            if h is None:
                hvp_list.append(torch.zeros_like(p).contiguous().view(-1))
            else:
                hvp_list.append(h.contiguous().view(-1))
        hvp_flat = torch.cat(hvp_list)

        return hvp_flat


    def estimate_max_eigenvalue(self, num_iterations=10, tolerance=1e-4):
        """
        Estimates the maximum eigenvalue (lambda_max) of the Hessian using 
        the Power Iteration method. This is a measure of sharpness.
        """
        # Initialize a random vector v
        param_shapes = [p.shape for p in self.model.parameters()]
        param_numel = [p.numel() for p in self.model.parameters()]
        total_params = sum(param_numel)
        
        v_flat = torch.randn(total_params, device=self.device)
        v_flat = v_flat / torch.norm(v_flat)
        
        # Unflatten v_flat into a list of tensors matching model structure
        def unflatten_vector(flat_vector, shapes, numel):
            vectors = []
            start = 0
            for shape, n in zip(shapes, numel):
                vectors.append(flat_vector[start:start+n].view(shape))
                start += n
            return vectors

        lambda_max = 0.0
        
        for i in range(num_iterations):
            v_unflat = unflatten_vector(v_flat, param_shapes, param_numel)
            
            # Compute H * v
            hvp_flat = self.hessian_vector_product(v_unflat)
            
            # Estimate eigenvalue: lambda = v^T * (H * v)
            new_lambda_max = torch.dot(v_flat, hvp_flat).item()
            
            # Check for convergence
            if i > 0 and abs(new_lambda_max - lambda_max) < tolerance:
                print(f"Power iteration converged after {i+1} iterations.")
                break
            
            lambda_max = new_lambda_max
            
            # Normalize v for the next iteration: v = (H * v) / ||H * v||
            v_flat = hvp_flat / torch.norm(hvp_flat)
            
        return lambda_max, v_flat

    def visualize_loss_landscape(self, direction_1_flat, direction_2_flat, steps=20, max_range=1.0):
        """
        Generates a 2D slice of the loss landscape along two directions.
        
        Args:
            direction_1_flat (torch.Tensor): Flattened vector for the first direction.
            direction_2_flat (torch.Tensor): Flattened vector for the second direction.
            steps (int): Number of steps along each direction.
            max_range (float): Maximum distance from the minimum in each direction.
            
        Returns:
            tuple: (X, Y, Z) where X, Y are coordinates and Z is the loss value.
        """
        
        # Ensure directions are normalized
        d1 = direction_1_flat / torch.norm(direction_1_flat)
        d2 = direction_2_flat / torch.norm(direction_2_flat)
        
        # Orthogonalize d2 w.r.t. d1 (Gram-Schmidt)
        d2 = d2 - torch.dot(d2, d1) * d1
        d2 = d2 / torch.norm(d2)
        
        # Get the current minimum (w*)
        w_star_flat = self._get_flat_params().detach().clone()
        
        # Create grid
        alphas = torch.linspace(-max_range, max_range, steps)
        betas = torch.linspace(-max_range, max_range, steps)
        
        X, Y = torch.meshgrid(alphas, betas, indexing='ij')
        Z = torch.zeros_like(X)
        
        # Unflatten w* and directions for setting model parameters
        param_shapes = [p.shape for p in self.model.parameters()]
        param_numel = [p.numel() for p in self.model.parameters()]
        
        def unflatten_vector(flat_vector, shapes, numel):
            vectors = []
            start = 0
            for shape, n in zip(shapes, numel):
                vectors.append(flat_vector[start:start+n].view(shape))
                start += n
            return vectors

        w_star_unflat = unflatten_vector(w_star_flat, param_shapes, param_numel)
        d1_unflat = unflatten_vector(d1, param_shapes, param_numel)
        d2_unflat = unflatten_vector(d2, param_shapes, param_numel)
        
        # Iterate over the grid
        for i in range(steps):
            for j in range(steps):
                alpha = alphas[i].item()
                beta = betas[j].item()
                
                # w = w* + alpha * d1 + beta * d2
                new_w_unflat = []
                for w_star_p, d1_p, d2_p in zip(w_star_unflat, d1_unflat, d2_unflat):
                    new_w_unflat.append(w_star_p + alpha * d1_p + beta * d2_p)
                
                # Set model parameters to new_w
                with torch.no_grad():
                    for p, new_p in zip(self.model.parameters(), new_w_unflat):
                        p.copy_(new_p)
                
                # Compute loss at w
                loss_value = self._get_full_loss()
                Z[i, j] = loss_value
                
        # Restore original weights
        with torch.no_grad():
            for p, w_star_p in zip(self.model.parameters(), w_star_unflat):
                p.copy_(w_star_p)
                
        return X.numpy(), Y.numpy(), Z.numpy()

# --- Example Usage (Conceptual) ---
if __name__ == '__main__':
    # 1. Setup a dummy model and data
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    model = SimpleNet()
    loss_fn = nn.MSELoss()
    
    # Dummy data
    X_data = torch.randn(100, 10)
    Y_data = torch.randn(100, 1)
    dataset = TensorDataset(X_data, Y_data)
    data_loader = DataLoader(dataset, batch_size=32)
    
    # 2. Train the model to a minimum (simulated)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(5):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
    # 3. Initialize the Prober
    prober = LossLandscapeProber(model, loss_fn, data_loader)
    
    # 4. Estimate Sharpness (Max Eigenvalue)
    print("Estimating maximum Hessian eigenvalue (Sharpness)...")
    lambda_max, max_eigenvector = prober.estimate_max_eigenvalue(num_iterations=20)
    print(f"Maximum Eigenvalue (Sharpness): {lambda_max:.4f}")
    
    # 5. Visualize Loss Landscape (Conceptual - requires plotting library like matplotlib)
    # For a second direction, we can use a random orthogonal vector or the smallest eigenvector
    # For simplicity, we'll use a random orthogonal vector here.
    
    # Generate a random vector and orthogonalize it to the max eigenvector
    total_params = max_eigenvector.numel()
    random_dir = torch.randn(total_params, device=prober.device)
    random_dir = random_dir - torch.dot(random_dir, max_eigenvector) * max_eigenvector
    random_dir = random_dir / torch.norm(random_dir)
    
    print("\nGenerating 2D loss landscape slice...")
    # X, Y, Z = prober.visualize_loss_landscape(max_eigenvector, random_dir, steps=10, max_range=0.1)
    # print(f"Loss landscape grid generated. Z shape: {Z.shape}")
    # print(f"Min Loss: {Z.min():.4f}, Max Loss: {Z.max():.4f}")
    
    # Note: Plotting requires a separate step and library (e.g., matplotlib)
    # The output here is just the data for plotting.
    
    # Restore original weights (already done inside visualize_loss_landscape, but good practice)
    # w_star_flat = prober._get_flat_params().detach().clone()
    # ... restore logic ...
    pass
