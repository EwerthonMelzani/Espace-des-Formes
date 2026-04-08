import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from sklearn.datasets import make_circles, make_moons, make_blobs, make_gaussian_quantiles
from sklearn.model_selection import train_test_split

class DiffeomorphicLearnerTorch(nn.Module):
    def __init__(self, n, d, c, T=8, rho=0.2, sigma2=1.0, l2=1e-2):
        super().__init__()
        self.T = T
        self.dt = 1.0 / T
        self.rho = rho
        self.sigma2 = sigma2
        self.l2 = l2
        self.A = nn.Parameter(torch.zeros(T, n, d))
        self.A_aff = nn.Parameter(torch.zeros(T, d, d))
        self.b_aff = nn.Parameter(torch.zeros(T, d))
        self.W = nn.Parameter(torch.randn(d, c) * 0.01)
        self.b = nn.Parameter(torch.zeros(c))
        
    def gaussian_kernel(self, X, Y):
        XX = (X ** 2).sum(dim=1, keepdim=True)
        YY = (Y ** 2).sum(dim=1, keepdim=True).t()
        dist = XX + YY - 2 * X @ Y.t()
        return torch.exp(-dist / (2 * self.rho ** 2))
        
    def forward_train(self, X):
        Z = X
        path = [Z.detach().numpy()]
        Ks = []
        for t in range(self.T):
            K = self.gaussian_kernel(Z, Z)
            Ks.append(K)
            v = Z @ self.A_aff[t].t() + self.b_aff[t].unsqueeze(0) + K @ self.A[t]
            Z = Z + self.dt * v
            path.append(Z.detach().numpy())
        return Z, Ks, path

    def forward_test(self, X_test, X_train):
        Z_train = X_train
        Z_test = X_test
        path = [Z_test.detach().numpy()]
        for t in range(self.T):
            K_train = self.gaussian_kernel(Z_train, Z_train)
            K_cross = self.gaussian_kernel(Z_test, Z_train)
            v_train = Z_train @ self.A_aff[t].t() + self.b_aff[t].unsqueeze(0) + K_train @ self.A[t]
            v_test = Z_test @ self.A_aff[t].t() + self.b_aff[t].unsqueeze(0) + K_cross @ self.A[t]
            Z_train = Z_train + self.dt * v_train
            Z_test = Z_test + self.dt * v_test
            path.append(Z_test.detach().numpy())
        return Z_test, path

    def get_loss(self, X, y):
        Z_final, Ks, _ = self.forward_train(X)
        logits = Z_final @ self.W + self.b
        ce_loss = nn.CrossEntropyLoss(reduction='sum')(logits, y) / self.sigma2
        reg_loss = self.l2 * torch.sum(self.W ** 2)
        deform_rkhs = 0.0
        for t in range(self.T):
            deform_rkhs += self.dt * torch.sum((Ks[t] @ self.A[t]) * self.A[t])
        deform_affine = 0.0
        for t in range(self.T):
            deform_affine += self.dt * (0.5 * torch.sum(self.A_aff[t] ** 2) + torch.sum(self.b_aff[t] ** 2))
        return ce_loss + reg_loss + deform_rkhs + deform_affine

def build_dataset(name, seed, n_samples=200):
    if name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=0.05, factor=0.5, random_state=seed)
    elif name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
    elif name == "blobs":
        X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2, random_state=seed)
    elif name == "gaussian_quantiles":
        X, y = make_gaussian_quantiles(n_samples=n_samples, n_features=2, n_classes=2, random_state=seed)
    return X.astype(float), y.astype(int)

def add_one_dimension_with_noise(X2, std=0.01, seed=0):
    rng = np.random.default_rng(seed)
    z = std * rng.normal(size=(len(X2), 1))
    return np.hstack([X2, z])

def estimate_rho_like_paper(X, y):
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    same_vals, other_vals = [], []
    for i in range(len(X)):
        same = D[i, y == y[i]]
        same = same[same > 0]
        if len(same): same_vals.append(np.percentile(same, 5))
        other = D[i, y != y[i]]
        if len(other): other_vals.append(np.min(other))
    rho1 = np.percentile(same_vals, 75)
    rho2 = np.percentile(other_vals, 10)
    return max(1e-3, min(rho1, rho2))

def train_and_eval(ds_name, seed, use_3d=True):
    torch.manual_seed(42)
    X2, y = build_dataset(ds_name, seed=seed)
    idx_tr, idx_te = train_test_split(np.arange(len(X2)), test_size=0.30, random_state=seed, stratify=y)
    
    if use_3d:
        X = add_one_dimension_with_noise(X2, std=0.01, seed=seed+1234)
    else:
        X = X2
        
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    
    rho = estimate_rho_like_paper(X_tr, y_tr)
    n, d = X_tr.shape
    c = int(y.max() + 1)
    
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    
    model = DiffeomorphicLearnerTorch(n, d, c, T=8, rho=rho, sigma2=1.0, l2=1e-2)
    optimizer = optim.Adam(model.parameters(), lr=2e-2)
    
    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        loss = model.get_loss(X_tr_t, y_tr_t)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        Z_test, path = model.forward_test(X_te_t, X_tr_t)
        logits = Z_test @ model.W + model.b
        preds = torch.argmax(logits, dim=1)
        acc = float((preds == torch.tensor(y_te)).float().mean())
        
    return acc, path, y_te

DATASET_NAMES = ["circles", "moons", "blobs", "gaussian_quantiles"]
SEEDS = [0, 1, 2]

os.makedirs('animacoes', exist_ok=True)

for ds in DATASET_NAMES:
    best_acc = -1
    best_seed = 0
    best_path = None
    best_y = None
    use_3d = True
    
    for s in SEEDS:
        acc, path, y_te = train_and_eval(ds, s, use_3d=use_3d)
        if acc > best_acc:
            best_acc = acc
            best_seed = s
            best_path = path
            best_y = y_te
            
    print(f"{ds} best seed: {best_seed} with acc {best_acc:.3f}")
    
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121, projection='3d' if use_3d else None)
    ax2 = fig.add_subplot(122, projection='3d' if use_3d else None)
    
    colors = ['red' if yi == 0 else 'blue' for yi in best_y]
    
    pts_start = best_path[0]
    if use_3d:
        ax1.scatter(pts_start[:,0], pts_start[:,1], pts_start[:,2], c=colors)
    else:
        ax1.scatter(pts_start[:,0], pts_start[:,1], c=colors)
    ax1.set_title("Before")
    
    pts_end = best_path[-1]
    if use_3d:
        ax2.scatter(pts_end[:,0], pts_end[:,1], pts_end[:,2], c=colors)
    else:
        ax2.scatter(pts_end[:,0], pts_end[:,1], c=colors)
    ax2.set_title("After")
    
    plt.savefig(f'animacoes/{ds}_before_after.png', dpi=150)
    plt.close()
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d' if use_3d else None)
    
    def update(frame):
        ax.clear()
        pts = best_path[frame]
        if use_3d:
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=colors)
            ax.set_zlim(np.min([p[:,2] for p in best_path]), np.max([p[:,2] for p in best_path]))
        else:
            ax.scatter(pts[:,0], pts[:,1], c=colors)
        ax.set_xlim(np.min([p[:,0] for p in best_path]), np.max([p[:,0] for p in best_path]))
        ax.set_ylim(np.min([p[:,1] for p in best_path]), np.max([p[:,1] for p in best_path]))
        ax.set_title(f"Transformation t={frame}/8")
        
    ani = animation.FuncAnimation(fig, update, frames=len(best_path), repeat=False)
    ani.save(f'animacoes/{ds}_anim.gif', writer='pillow', fps=4)
    plt.close()

print('Finished generating animations and plots.')
