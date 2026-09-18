import numpy as np
from adam import Adam

class Perceptron:
    def __init__(self, input_size, hidden_size=5, seed=42):
        """
        input_size - кол-во признаков
        hidden_size - кол-во нейронов в скрытом слое
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rng = np.random.default_rng(seed)
        self.w_1 = self.rng.normal(loc=0, scale=0.1, size=(hidden_size, input_size))
        self.w_2 = self.rng.normal(loc=0, scale=0.1, size=(1, hidden_size))
        self.b_1 = np.zeros(hidden_size)
        self.b_2 = np.zeros(1)
        self.y_pred = None

        self.z_1 = None
        self.a_1 = None
        self.z_2 = None

    def calculate_metrics(self, y, y_pred):
        y = y.ravel()
        y_pred_class = (y_pred >= 0.5).astype(int).ravel()
    
        loss = self.loss(y, y_pred)
    
        accuracy = np.mean(y_pred_class == y)
    
        tp = np.sum((y == 1) & (y_pred_class == 1))
        tn = np.sum((y == 0) & (y_pred_class == 0))
        fp = np.sum((y == 0) & (y_pred_class == 1))
        fn = np.sum((y == 1) & (y_pred_class == 0))
    
        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )
    
        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )
    
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
    
        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    @classmethod
    def hyperparameter_validation(cls, X_train, y_train, X_val, y_val, hidden_sizes, learning_rates, batch_sizes, epochs=100):
        results = []

        for h_s in hidden_sizes:
            for lr in learning_rates:
                for b_s in batch_sizes:
                    model = cls(input_size=X_train.shape[1], hidden_size=h_s)
    
                    optimizer = Adam(lr=lr, parameters=model.parameters())
    
                    model.fit(
                        X_train,
                        y_train,
                        optimizer=optimizer,
                        batch_size=b_s,
                        epochs=epochs
                    )

                    val_pred = model.predict_proba(X_val)

                    val_metrics = model.calculate_metrics(
                        y_val,
                        val_pred
                    )
        
                    results.append({
                        "hidden_size": h_s,
                        "lr": lr,
                        "batch_size": b_s,
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                        "val_f1": val_metrics["f1"]
                    })
        
        return results

    def forward(self, X):
        """
        X -> z_1 = W_1 * X + b_1 -> a_1 = BReLU(z_1) -> z_2 = W_2 * a_1 + b_2 -> Y = Sigmoid(z_2)
        """
        def brelu(x):
            return np.clip(x, 0, 1)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # т.е. левое слагаемое - матрица в которой строка - объект на разных нейронах
        # столбец - признак на разных нейронах
        # поэтому прибавляем сдвиг к столбцам
        self.z_1 = X @ self.w_1.T + self.b_1

        self.a_1 = brelu(self.z_1)

        self.z_2 = self.a_1 @ self.w_2.T + self.b_2

        self.y_pred = sigmoid(self.z_2)

        return self.y_pred

    def backward(self, X, y):
        y = y.reshape(-1, 1)
        
        # ошибка выходного слоя
        delta_2 = self.y_pred - y

        grad_w_2 = delta_2.T @ self.a_1 / X.shape[0]

        grad_b_2 = np.mean(delta_2)

        grad_a_1 = delta_2 @ self.w_2
        brelu_grad = ((self.z_1 > 0) & (self.z_1 < 1))
        delta_1 = grad_a_1 * brelu_grad

        grad_w_1 = delta_1.T @ X / X.shape[0]

        grad_b_1 = np.mean(delta_1, axis=0)

        return [grad_w_1, grad_b_1, grad_w_2, grad_b_2]

    def loss(self, y, y_pred):
        y = y.reshape(-1, 1)
        eps = 1e-15

        y_pred = np.clip(y_pred, eps, 1 - eps)

        return -np.mean(
            y * np.log(y_pred) +
            (1 - y) * np.log(1 - y_pred)
        )

    def fit(self, X_train, y_train, optimizer, batch_size=32, epochs=1000, X_val=None, y_val=None, print_epoch_k=None):
        metrics_history = {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }

        n_samples = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            # перемешиваем данные
            indices = self.rng.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for start in range(0, n_samples, batch_size):
                # Достаем батч
                end = start + batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                self.y_pred = self.forward(X_batch)

                grads = self.backward(X_batch, y_batch)

                optimizer.step(grads)

            train_pred = self.forward(X_train)

            metrics = self.calculate_metrics(
                y_train,
                train_pred
            )
    
            for metric_name in metrics_history:
                metrics_history[metric_name].append(
                    metrics[metric_name]
                )

            if print_epoch_k is not None and epoch % print_epoch_k == 0:
                print(f"Epoch: {epoch}, loss: {metrics['loss']:.4f}, f1: {metrics['f1']:.4f}")

        return metrics_history
                
    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def parameters(self):
        return [
            self.w_1,
            self.b_1,
            self.w_2,
            self.b_2
        ]