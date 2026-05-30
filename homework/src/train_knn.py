from sklearn.neighbors import KNeighborsRegressor

from ._internals.calculate_metrics import calculate_metrics
from ._internals.prepare_data import prepare_data
from ._internals.print_metrics import print_metrics
from ._internals.run_experiment import run_experiment
from ._internals.save_model import save_model

x_train, x_test, y_train, y_test = prepare_data()

# entrenar el modelo
estimator = KNeighborsRegressor(n_neighbors=5)
estimator.fit(x_train, y_train)
save_model(estimator)

print()
print(estimator, ":", sep="")

# Metricas de error durante entrenamiento
mse, mae, r2 = calculate_metrics(x_train, y_train, estimator)
print_metrics(mse, mae, r2, title="Metricas de entrenamiento:")

# Metricas de error durante testing
mse, mae, r2 = calculate_metrics(x_test, y_test, estimator)
print_metrics(mse, mae, r2, title="Metricas de testing:")
