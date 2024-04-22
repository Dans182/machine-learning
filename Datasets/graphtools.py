# graphtools v1.0 27/Junio/2023

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

colors = ListedColormap(["#E67332", "#18B95E", "#326DE6"])



def _plot_decision_boundaries(model, X_train, X_test, ax, n_points = (500, 500)):
    """Muestra las fronteras de decisión de un modelo de clasificación:
    * model: Modelo entrenado
    * X_train: Muestras de entrenamiento
    * X_test: Muestras de validación
    * ax: Conjunto de ejes de matplotlib en el que generar la imagen
    * n_points: Número de puntos del grid a generar (ancho x alto)
    """
    if not X_test is None:                                              # Unimos ambas estructuras
        X = np.concatenate([X_train, X_test])
    else:
        X = X_train
    maxX = max(X[:, 0])                                                 # Valores extremos de los datos
    minX = min(X[:, 0])
    maxY = max(X[:, 1])
    minY = min(X[:, 1])
    marginX = (maxX - minX) * 0.1                                       # Añadimos un 10% de margen
    marginY = (maxY - minY) * 0.1
    x = np.linspace(minX - marginX, maxX + marginX, n_points[0])
    y = np.linspace(minY - marginY, maxY + marginY, n_points[1])
    X, Y = np.meshgrid(x, y)
    Z = model.predict(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape).round().astype("int")
    ax.contourf(X, Y, Z, levels = 2,
        colors = ["#E3BCAB", "#B0D9CB", "#75B6E6"],
        zorder = 0
    )



def show_boundaries(model, X_train, y_train, X_test = None, y_test = None, labels = [], n_points = (500, 500),
            size = (9, 7), show_scatter = True, s = 25, show = True, axis = None, aspect = "auto"):
    """Muestra las fronteras de decisión de un modelo de clasificación y,
    sobre él, las muestras de entrenamiento y de validación:
    * model: Modelo entrenado
    * X_train: Características de entrenamiento (array o DataFrame)
    * y_train: Etiquetas de entrenamiento
    * X_test: Características de validación (array o DataFrame)
    * y_test: Etiquetas de validación
    * labels: Identificadores de las clases
    * n_points: Número de puntos del grid a generar (ancho x alto)
    * size: Ancho y alto en pulgadas de la imagen a generar
    * show_scatter: Si toma el valor True, se mostrará el scatter plot sobre el mapa de fronteras
    * show: Si toma el valor True, se ejecutará la función plt.show()
    * axis: Eje en el que crear la imagen
    """
    if isinstance(X_train, pd.core.frame.DataFrame):                                    # Convertimos las estructuras a arrays NumPy
        X_train = X_train.values
    if isinstance(X_test, pd.core.frame.DataFrame):
        X_test = X_test.values
    if axis == None:                                                                    # Si no se ha indicado el eje en el que crear la gráfica
        fig, ax = plt.subplots(figsize = size)                                          # creamos una figura y un conjunto de ejes
        ax.set_aspect(aspect)
    else:
        ax = axis
    if model != None:                                                                   # Creamos el mapa con las fronteras de decisión
        _plot_decision_boundaries(model, X_train, X_test, ax, n_points = n_points)
    # Train dataset
    if show_scatter:                                                                    # Mostramos el diagrama de dispersión
        scatter = ax.scatter(
            x = X_train[:, 0], y = X_train[:, 1], c = y_train, s = s,
            cmap = colors, zorder = 2, edgecolor = "#666666", linewidths = 0.7
        )
        # Test dataset
    if (not(X_test is None)) and show_scatter:
        scatter = plt.scatter(
            x = X_test[:, 0], y = X_test[:, 1], c = y_test, s = s,
            cmap = colors, zorder = 2, edgecolor = "#FFFFFF", linewidths = 0.7
        )
    if (len(labels) > 0) and show_scatter:                                              # Si se han pasado etiquetas, se muestra la leyenda
        ax.legend(
            handles = scatter.legend_elements()[0],
            labels = list(labels)
        )

    ax.grid(color = "#EEEEEE", zorder = 1, alpha = 0.4)
    if show:                                                                            # Si hay que ejecutar la función plt.show(), se ejecuta
        plt.show()
    elif (not show) and (axis == None):
        return fig, ax




def scatter3D(model, X_train, y_train, X_test = None, y_test = None, labels = [], n_points = (25, 25, 25),
            size = (9, 7), show_boundaries = True, show_scatter = True, s = 20, show = True, axis = None, aspect = "auto"):
    """Muestra las fronteras de decisión de un modelo de clasificación y,
    sobre él, las muestras de entrenamiento y de validación:
    * model: Modelo entrenado
    * X_train: Características de entrenamiento (array o DataFrame)
    * y_train: Etiquetas de entrenamiento
    * X_test: Características de validación (array o DataFrame)
    * y_test: Etiquetas de validación
    * labels: Identificadores de las clases
    * n_points: Número de puntos del grid a generar (ancho x alto)
    * size: Ancho y alto en pulgadas de la imagen a generar
    * show_scatter: Si toma el valor True, se mostrará el scatter plot sobre el mapa de fronteras
    * show: Si toma el valor True, se ejecutará la función plt.show()
    * axis: Eje en el que crear la imagen
    """
    if isinstance(X_train, pd.core.frame.DataFrame):                                    # Convertimos las estructuras a arrays NumPy
        X_train = X_train.values
    if isinstance(X_test, pd.core.frame.DataFrame):
        X_test = X_test.values
    if axis == None:                                                                    # Si no se ha indicado el eje en el que crear la gráfica
        fig, ax = plt.subplots(figsize = size, subplot_kw = {"projection": "3d"})       # creamos una figura y un conjunto de ejes
        ax.set_aspect(aspect)
    if X_train.shape[1] > 3:
            pca = PCA(n_components = 3)
            X_train = pca.fit_transform(X_train)
            if not X_test is None:
                X_test = pca.transform(X_test)
    # Train dataset
    if show_scatter:                                                                    # Mostramos el diagrama de dispersión
        scatter = ax.scatter3D(
            xs = X_train[:, 0],
            ys = X_train[:, 1],
            zs = X_train[:, 2],
            c = y_train, s = s,
            cmap = colors, zorder = 2, edgecolor = "#666666", linewidths = 0.6
        )
        # Test dataset
    if (not(X_test is None)) and show_scatter:
        scatter = plt.scatter3D(
            xs = X_test[:, 0],
            ys = X_test[:, 1],
            zs = X_test[:, 2],
            c = y_test, s = s,
            cmap = colors, zorder = 2, edgecolor = "#FFFFFF", linewidths = 0.6
        )
    if (len(labels) > 0) and show_scatter:                                              # Si se han pasado etiquetas, se muestra la leyenda
        ax.legend(
            handles = scatter.legend_elements()[0],
            labels = list(labels)
        )

    ax.grid(color = "#EEEEEE", zorder = 1, alpha = 0.4)
    if show:                                                                            # Si hay que ejecutar la función plt.show(), se ejecuta
        plt.show()
    elif (not show) and (axis == None):
        return fig, ax



def show_digits(X, y, nrows, ncols, prediction = None):
    """ Muestra una matriz de dígitos resaltando aquellos para los que la predicción es incorrecta """
    if isinstance(X, pd.core.frame.DataFrame):                                    # Convertimos las estructuras a arrays NumPy
        X = X.values
    if prediction is None:
        prediction = []
    y = list(y)
    fig, ax = plt.subplots(nrows, ncols)
    fig.set_size_inches(17, np.ceil(nrows * 21 / ncols ))
    ax = ax.flatten()
    for i in range(ncols * nrows):
        cmap = "gray"
        title = str(y[i])
        if len(prediction) > 0:
            if y[i] != prediction[i]:
                title = title + " (" + str(prediction[i]) + ")"
                cmap = "Reds"
        ax[i].set_title(title)
        ax[i].imshow(X[i].reshape(28, 28), cmap = cmap);
        ax[i].set_xticks([])
        ax[i].set_yticks([])



def show_clothes(X, y, nrows, ncols, prediction = None):
    """ Muestra una matriz de prendas de vestir resaltando aquellos para los que la predicción es incorrecta """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    if isinstance(X, pd.core.frame.DataFrame):                                    # Convertimos las estructuras a arrays NumPy
        X = X.values
    if prediction is None:
        prediction = []
    y = list(y)
    fig, ax = plt.subplots(nrows, ncols)
    fig.set_size_inches(17, np.ceil(nrows * 21 / ncols ))
    ax = ax.flatten()
    for i in range(ncols * nrows):
        cmap = "gray_r"
        title = str(class_names[y[i]])
        if len(prediction) > 0:
            y_pred = prediction[i].argmax()
            if y[i] != y_pred:
                title = title + " (" + class_names[y_pred] + ")"
                cmap = "Reds"
        ax[i].set_title(title)
        ax[i].imshow(X[i], cmap = cmap);
        ax[i].set_xticks([])
        ax[i].set_yticks([])