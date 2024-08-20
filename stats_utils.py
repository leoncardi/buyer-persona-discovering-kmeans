import matplotlib.pyplot as plt

def elbow_method(model_class, X, max_clusters=10, **model_kwargs):
    """
    Executa o método do cotovelo para determinar o número ideal de clusters.

    Parâmetros:
    - model_class: Classe do modelo de clustering (ex: KMeans, KPrototypes).
    - X: Dataset para treinamento.
    - max_clusters: Número máximo de clusters a testar.
    - model_kwargs: Argumentos adicionais para o modelo (ex: init, random_state, etc.).
    
    Retorna:
    - distortions: Lista de distorções (custo) para cada número de clusters.
    """
    distortions = []
    K = range(1, max_clusters + 1)
    
    for k in K:
        # Cria o modelo com o número de clusters
        model_kwargs['n_clusters'] = k
        model = model_class(**model_kwargs)
        
        # Ajusta o modelo
        if 'categorical' in model_kwargs:
            clusters = model.fit_predict(X, categorical=model_kwargs['categorical'])
        else:
            clusters = model.fit_predict(X)
        
        # Obtém a distorção/custo
        distortions.append(model.cost_ if hasattr(model, 'cost_') else model.inertia_)
    
    # Plotando o gráfico do método do cotovelo
    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Custo (Distorção)')
    plt.title('Método do Cotovelo para Determinar o Número Ótimo de Clusters')
    plt.show()

    return distortions

"""
    # Exemplo de uso para K-Means
    from sklearn.cluster import KMeans
    elbow_method(KMeans, X, max_clusters=10, init='k-means++', random_state=42)

    # Exemplo de uso para K-Prototypes
    from kmodes.kprototypes import KPrototypes
    categorical_indices = get_categorical_indices(X)
    elbow_method(KPrototypes, X, max_clusters=10, init='Cao', categorical=categorical_indices, random_state=42)
"""