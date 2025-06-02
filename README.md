# Trabalho de Clusterização e KNN

Este projeto foi desenvolvido para a disciplina **Tópicos em Engenharia de Software** da Unicesumar, a pedido do professor **Ricardo Satin**.

## Descrição do Trabalho

O objetivo deste trabalho é implementar um processo completo de clusterização utilizando a técnica **K-means**, abordando todas as etapas do algoritmo, desde a manipulação dos dados até a criação e reorganização dos clusters. Além disso, o projeto prepara a base de dados para uma futura aplicação do algoritmo **K-Nearest Neighbors (KNN)**, incluindo suporte para atributos categóricos sem a necessidade de codificação prévia.

O trabalho abrange:

- Implementação da estrutura de dados para registros e clusters;
- Atribuição de elementos a clusters com base na distância euclidiana;
- Recalculo e atualização dos centróides dos clusters;
- Análise de dispersão para criação de novos clusters quando necessário;
- Transformação dinâmica de atributos categóricos para uso em algoritmos numéricos;
- Implementação do algoritmo KNN com suporte a atributos categóricos não codificados;
- KNN ponderado para considerar a proximidade dos vizinhos na votação;
- Separação do conjunto de dados em treinamento e validação para avaliação do modelo.

## Tecnologias Utilizadas

- Python 3
- Bibliotecas padrão (math, typing, collections, random)

## Como usar

1. Clone este repositório.
2. Execute o arquivo principal `cluster_knn.py` para rodar as funções de clusterização e KNN.
3. Os dados de exemplo estão incluídos no código para demonstração das funcionalidades.
