from collections import Counter
from random import shuffle
from math import sqrt
from typing import List, Tuple

# ====== REGISTRO ======

class Registro:
    def __init__(self, id, atributos_numericos: List[float], atributos_categoricos: List[str] = None):
        self.id = id
        self.atributos_numericos = atributos_numericos
        self.atributos_categoricos = atributos_categoricos or []

    def __repr__(self):
        return f'Registro({self.id}, Num={self.atributos_numericos}, Cat={self.atributos_categoricos})'


# ====== DISTÂNCIAS ======

def distancia_euclidiana(v1, v2):
    return sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

def distancia_hamming(cat1, cat2):
    return sum(a != b for a, b in zip(cat1, cat2))

def distancia_total(r1: Registro, r2: Registro, peso_num=1.0, peso_cat=1.0):
    """Distância combinada numérica e categórica."""
    dist_num = distancia_euclidiana(r1.atributos_numericos, r2.atributos_numericos) if r1.atributos_numericos else 0
    dist_cat = distancia_hamming(r1.atributos_categoricos, r2.atributos_categoricos) if r1.atributos_categoricos else 0
    return peso_num * dist_num + peso_cat * dist_cat


# ====== KNN ======

def knn_classificar(novo_registro: Registro, registros_treinamento: List[Registro], k: int = 3, ponderado: bool = False) -> str:
    """Classifica um registro novo usando KNN com atributos categóricos não codificados e ponderação opcional."""
    distancias = []
    for r in registros_treinamento:
        dist = distancia_total(novo_registro, r)
        distancias.append((dist, r))

    # Ordena pelas menores distâncias
    distancias.sort(key=lambda x: x[0])
    k_vizinhos = distancias[:k]

    # Votação ponderada ou simples
    votos = {}
    for dist, r in k_vizinhos:
        classe = r.atributos_categoricos[0] if r.atributos_categoricos else 'Desconhecido'
        peso = 1 / (dist + 1e-5) if ponderado else 1
        votos[classe] = votos.get(classe, 0) + peso

    # Classe com maior peso
    mais_comum = max(votos.items(), key=lambda x: x[1])

    return mais_comum[0]


# ====== SEPARAÇÃO DE DADOS ======

def separar_dados(lista: List[Registro], proporcao_treinamento: float = 0.7) -> Tuple[List[Registro], List[Registro]]:
    """Separa a lista em treinamento e validação."""
    lista_copy = lista[:]
    shuffle(lista_copy)
    corte = int(len(lista_copy) * proporcao_treinamento)
    return lista_copy[:corte], lista_copy[corte:]


# ====== USO ======

if __name__ == "__main__":
    # Base de registros
    registros = [
        Registro(1, [1.0, 2.0], ["ClasseA"]),
        Registro(2, [1.5, 1.8], ["ClasseA"]),
        Registro(3, [5.0, 8.0], ["ClasseB"]),
        Registro(4, [6.0, 9.0], ["ClasseB"]),
        Registro(5, [1.2, 0.9], ["ClasseA"]),
        Registro(6, [5.5, 8.5], ["ClasseB"]),
        Registro(7, [1.3, 1.0], ["ClasseA"]),
        Registro(8, [6.2, 8.9], ["ClasseB"]),
    ]

    # Separar em treinamento e validação
    treinamento, validacao = separar_dados(registros, proporcao_treinamento=0.75)

    print("Treinamento:")
    for r in treinamento:
        print(r)

    print("\nValidação:")
    for r in validacao:
        print(r)

    # Classificar registros de validação
    print("\nResultados da validação:")
    for r in validacao:
        classe_predita = knn_classificar(r, treinamento, k=3, ponderado=True)
        print(f"Registro {r.id} - Classe real: {r.atributos_categoricos[0]} - Classe predita: {classe_predita}")
