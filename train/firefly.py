import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from models.classifier import MLPClassifier

from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


################ skill issue code ##################################

# firefly.py  — parallel version
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm

from models.classifier import MLPClassifier


# ── Must be module-level (not inside a class) so multiprocessing can pickle it ──
def _svm_fitness_worker(args):
    """Standalone function evaluated in a worker process."""
    X_train, y_train, X_val, y_val, mask = args
    if np.sum(mask) == 0:
        return 0.0
    X_tr = X_train[:, mask == 1]
    X_v  = X_val[:, mask == 1]
    clf  = SVC(kernel='linear')
    clf.fit(X_tr, y_train)
    return accuracy_score(y_val, clf.predict(X_v))


class FireflyFeatureSelectionSVM:
    def __init__(self, n_fireflies, n_features, alpha=0.5, beta0=1, gamma=1,
                 max_iter=20, n_workers=None):
        self.n_fireflies = n_fireflies
        self.n_features  = n_features
        self.alpha       = alpha
        self.beta0       = beta0
        self.gamma       = gamma
        self.max_iter    = max_iter
        # None → use all CPU cores automatically
        self.n_workers   = n_workers or os.cpu_count()

    def initialize_population(self):
        return np.random.randint(0, 2, (self.n_fireflies, self.n_features))

    def _eval_population_parallel(self, fireflies, X_train, y_train, X_val, y_val):
        """Evaluate the whole population in parallel."""
        args = [(X_train, y_train, X_val, y_val, mask) for mask in fireflies]
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(_svm_fitness_worker, args))
        return np.array(results)

    def move_firefly(self, xi, xj, beta):
        step = beta * (xj - xi) + self.alpha * (np.random.rand(self.n_features) - 0.5)
        prob = 1 / (1 + np.exp(-step))
        return (np.random.rand(self.n_features) < prob).astype(int)

    def run(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y
        )

        print("Evaluating initial population...")
        fireflies = self.initialize_population()

        fitness_vals = self._eval_population_parallel(
            fireflies, X_train, y_train, X_val, y_val
        )

        best_global = np.max(fitness_vals)

        outer_pbar = tqdm(range(self.max_iter), desc="Iterations")

        for t in outer_pbar:
            outer_pbar.set_postfix({"best": f"{best_global:.4f}"})

            # 🔥 adaptive randomness
            alpha_t = self.alpha * (0.97 ** t)

            new_fireflies = fireflies.copy()
            moves_needed = []

            # 🔥 tqdm for inner movement loop
            inner_pbar = tqdm(range(self.n_fireflies), leave=False, desc="Moving fireflies")

            for i in inner_pbar:
                xi = fireflies[i].copy()

                for j in range(self.n_fireflies):
                    if fitness_vals[j] > fitness_vals[i]:

                        r = np.sum(xi != fireflies[j])
                        beta = self.beta0 * np.exp(-self.gamma * r ** 2)

                        # movement
                        step = beta * (fireflies[j] - xi) + alpha_t * (np.random.rand(self.n_features) - 0.5)
                        prob = 1 / (1 + np.exp(-step))
                        xi = (np.random.rand(self.n_features) < prob).astype(int)

                # 🔥 mutation
                if np.random.rand() < 0.1:
                    idx = np.random.randint(self.n_features)
                    xi[idx] ^= 1

                new_fireflies[i] = xi
                moves_needed.append(i)

            # 🔥 parallel evaluation with tqdm
            moved_indices = list(set(moves_needed))
            moved_masks = new_fireflies[moved_indices]

            eval_pbar = tqdm(total=len(moved_masks), desc="Evaluating", leave=False)

            new_fit_values = []
            args = [(X_train, y_train, X_val, y_val, mask) for mask in moved_masks]

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                for res in executor.map(_svm_fitness_worker, args):
                    new_fit_values.append(res)
                    eval_pbar.update(1)

            eval_pbar.close()

            # 🔥 update only if improved
            for idx, fit in zip(moved_indices, new_fit_values):
                if fit > fitness_vals[idx]:
                    fireflies[idx] = new_fireflies[idx]
                    fitness_vals[idx] = fit

            # 🔥 diversity injection
            if t % 5 == 0:
                rand_idx = np.random.randint(self.n_fireflies)
                fireflies[rand_idx] = np.random.randint(0, 2, self.n_features)
                fitness_vals[rand_idx] = self._eval_population_parallel(
                    [fireflies[rand_idx]], X_train, y_train, X_val, y_val
                )[0]

            best_global = np.max(fitness_vals)

        best_index = np.argmax(fitness_vals)
        return fireflies[best_index], fitness_vals[best_index]

    # commented out cuz premature convergence 0.9400
    # def run(self, X, y):
    #     X_train, X_val, y_train, y_val = train_test_split(
    #         X, y, test_size=0.2, stratify=y
    #     )

    #     print("Evaluating initial population...")
    #     fireflies    = self.initialize_population()
    #     fitness_vals = self._eval_population_parallel(
    #         fireflies, X_train, y_train, X_val, y_val
    #     )

    #     for t in range(self.max_iter):
    #         print(f"\nIteration {t+1}/{self.max_iter}  |  "
    #               f"Best so far: {np.max(fitness_vals):.4f}")

    #         # Collect all moves that need a fitness re-evaluation
    #         moves_needed = []   # (i, new_mask)
    #         new_fireflies = fireflies.copy()

    #         for i in range(self.n_fireflies):
    #             for j in range(self.n_fireflies):
    #                 if fitness_vals[j] > fitness_vals[i]:
    #                     r    = np.sum(fireflies[i] != fireflies[j])
    #                     beta = self.beta0 * np.exp(-self.gamma * r ** 2)
    #                     new_fireflies[i] = self.move_firefly(
    #                         fireflies[i], fireflies[j], beta
    #                     )
    #                     moves_needed.append(i)
    #                     break   # only move toward the best attractor per firefly

    #         # Evaluate only the fireflies that actually moved — in parallel
    #         moved_indices  = list(set(moves_needed))
    #         moved_masks    = new_fireflies[moved_indices]
    #         new_fit_values = self._eval_population_parallel(
    #             moved_masks, X_train, y_train, X_val, y_val
    #         )

    #         for idx, fit in zip(moved_indices, new_fit_values):
    #             fireflies[idx]    = new_fireflies[idx]
    #             fitness_vals[idx] = fit

    #         print(f"Best Accuracy: {np.max(fitness_vals):.4f}")

    #     best_index = np.argmax(fitness_vals)
    #     return fireflies[best_index], fitness_vals[best_index]
    
######################################################################


# original
# class FireflyFeatureSelectionSVM:
#     def __init__(self, n_fireflies, n_features, alpha=0.5, beta0=1, gamma=1, max_iter=20):
#         self.n_fireflies = n_fireflies
#         self.n_features = n_features
#         self.alpha = alpha
#         self.beta0 = beta0
#         self.gamma = gamma
#         self.max_iter = max_iter

#     def initialize_population(self):
#         return np.random.randint(0, 2, (self.n_fireflies, self.n_features))

#     def fitness(self, X_train, y_train, X_val, y_val, mask):
#         if np.sum(mask) == 0:
#             return 0
#         X_train_sel = X_train[:, mask == 1]
#         X_val_sel = X_val[:, mask == 1]
#         clf = SVC(kernel='linear')
#         clf.fit(X_train_sel, y_train)
#         preds = clf.predict(X_val_sel)
#         return accuracy_score(y_val, preds)

#     def move_firefly(self, xi, xj, beta):
#         step = beta * (xj - xi) + self.alpha * (np.random.rand(self.n_features) - 0.5)
#         prob = 1 / (1 + np.exp(-step))
#         new_mask = (np.random.rand(self.n_features) < prob).astype(int)
#         return new_mask

#     def run(self, X, y):
#         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

#         # Initial population fitness calculation with progress bar
#         print("Evaluating initial population...")
#         fireflies = self.initialize_population()
#         fitness_vals = np.array([
#             self.fitness(X_train, y_train, X_val, y_val, mask)
#             for mask in tqdm(fireflies, desc="Initial Population")
#         ])

#         for t in range(self.max_iter):
#             pbar = tqdm(total=self.n_fireflies * self.n_fireflies, desc=f"Iteration {t+1}/{self.max_iter}")
#             for i in range(self.n_fireflies):
#                 for j in range(self.n_fireflies):
#                     if fitness_vals[j] > fitness_vals[i]:
#                         r = np.sum(fireflies[i] != fireflies[j])
#                         beta = self.beta0 * np.exp(-self.gamma * r ** 2)
#                         fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], beta)
#                         fitness_vals[i] = self.fitness(X_train, y_train, X_val, y_val, fireflies[i])
#                     pbar.update(1)
#             pbar.close()

#             print(f"Best Accuracy so far: {np.max(fitness_vals):.4f}")

#         best_index = np.argmax(fitness_vals)
#         return fireflies[best_index], fitness_vals[best_index]



class FireflyFeatureSelectionMLP:
    def __init__(self, n_fireflies, n_features, alpha=0.5, beta0=1, gamma=1, max_iter=20, device="cuda", mlp_epoch=10):
        self.n_fireflies = n_fireflies
        self.n_features = n_features
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.max_iter = max_iter
        self.device = device
        self.mlp_epoch = mlp_epoch

    def initialize_population(self):
        return np.random.randint(0, 2, (self.n_fireflies, self.n_features))

    def fitness(self, X_train, y_train, X_val, y_val, mask):
        if np.sum(mask) == 0:
            return 0

        # Select masked features
        X_train_sel = X_train[:, mask == 1]
        X_val_sel = X_val[:, mask == 1]

        # Convert to tensors
        X_train_t = torch.tensor(X_train_sel, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(self.device)
        X_val_t = torch.tensor(X_val_sel, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(self.device)

        # Create model
        model = MLPClassifier(X_train_sel.shape[1], len(np.unique(y_train))).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train small MLP (few epochs for speed)
        model.train()
        for _ in range(self.mlp_epoch):  # small number of epochs
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = model(X_val_t).argmax(dim=1)
            acc = (preds == y_val_t).float().mean().item()

        return acc

    def move_firefly(self, xi, xj, beta):
        step = beta * (xj - xi) + self.alpha * (np.random.rand(self.n_features) - 0.5)
        prob = 1 / (1 + np.exp(-step))
        new_mask = (np.random.rand(self.n_features) < prob).astype(int)
        return new_mask

    def run(self, X, y):
        # Scale features for better MLP convergence
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

        fireflies = self.initialize_population()
        fitness_vals = np.array([self.fitness(X_train, y_train, X_val, y_val, mask) for mask in tqdm(fireflies, desc="Initial Population")])

        for t in range(self.max_iter):
            pbar = tqdm(total=self.n_fireflies * self.n_fireflies, desc=f"Iteration {t+1}/{self.max_iter}")
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if fitness_vals[j] > fitness_vals[i]:
                        r = np.sum(fireflies[i] != fireflies[j])
                        beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                        fireflies[i] = self.move_firefly(fireflies[i], fireflies[j], beta)
                        fitness_vals[i] = self.fitness(X_train, y_train, X_val, y_val, fireflies[i])
                    pbar.update(1)
            pbar.close()

            print(f"Best Accuracy so far: {np.max(fitness_vals):.4f}")

        best_index = np.argmax(fitness_vals)
        return fireflies[best_index], fitness_vals[best_index]
