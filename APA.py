import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ==============================================================================
# 1. FESTA DO STUART (Maximum Weight Independent Set on a Tree)
# ==============================================================================

@dataclass
class Funcionario:
    """Representa um nó na árvore da hierarquia da empresa."""
    nome: str
    convivencia: int
    subordinados: List['Funcionario'] = field(default_factory=list)


def resolver_festa_stuart(funcionario: Funcionario) -> Tuple[int, int]:
    """
    Calcula a convivência máxima para a subárvore enraizada no funcionário.

    Retorna uma tupla (melhor_com_ele, melhor_sem_ele):
      - melhor_com_ele: soma máxima incluindo este funcionário (filhos não vão).
      - melhor_sem_ele: soma máxima excluindo este funcionário (filhos podem ir ou não).
    """
    # 1. Se eu vou, minha convivência conta.
    conv_com_ele = funcionario.convivencia

    # 2. Se eu não vou, começo com 0.
    conv_sem_ele = 0

    # Processa recursivamente (Bottom-Up)
    for sub in funcionario.subordinados:
        sub_com, sub_sem = resolver_festa_stuart(sub)

        # Se eu vou, meus filhos NÃO podem ir.
        conv_com_ele += sub_sem

        # Se eu NÃO vou, pego o melhor cenário de cada filho (ir ou não ir).
        conv_sem_ele += max(sub_com, sub_sem)

    return conv_com_ele, conv_sem_ele


# ==============================================================================
# 2. MULTIPLICAÇÃO DE CADEIAS DE MATRIZES (Matrix Chain Multiplication)
# ==============================================================================

def matrix_chain_order(dimensoes: List[int]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Calcula o custo mínimo de multiplicações.
    Entrada: dimensões = [p0, p1, ..., pn] onde a Matriz A_i é p_{i-1} x p_i.
    Retorna: (Tabela de Custos, Tabela de Cortes 'k')
    """
    n = len(dimensoes) - 1  # Número de matrizes
    # m[i][j] guarda o custo mínimo para multiplicar A_i..A_j
    m = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
    # s[i][j] guarda o índice k onde ocorreu o corte ótimo
    s = [[0 for _ in range(n + 1)] for _ in range(n + 1)]

    # l é o comprimento da cadeia sendo considerada (2 matrizes, depois 3...)
    for l in range(2, n + 1):
        for i in range(1, n - l + 2):
            j = i + l - 1
            m[i][j] = sys.maxsize  # Infinito

            # Testa todos os pontos de corte k entre i e j
            for k in range(i, j):
                # Custo = custo esq + custo dir + custo da fusão
                q = m[i][k] + m[k + 1][j] + (dimensoes[i - 1] * dimensoes[k] * dimensoes[j])

                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k

    return m, s


def reconstruir_parenteses(s: List[List[int]], i: int, j: int) -> str:
    """Função auxiliar para imprimir a parentesiação ótima."""
    if i == j:
        return f"A{i}"
    else:
        k = s[i][j]
        esq = reconstruir_parenteses(s, i, k)
        dir = reconstruir_parenteses(s, k + 1, j)
        return f"({esq} x {dir})"


# ==============================================================================
# 3. MAIOR SUBSEQUÊNCIA COMUM (Longest Common Subsequence - LCS)
# ==============================================================================

def lcs(texto1: str, texto2: str) -> str:
    """Encontra a maior subsequência comum entre duas strings."""
    m, n = len(texto1), len(texto2)

    # dp[i][j] guarda o tamanho da LCS entre texto1[0..i-1] e texto2[0..j-1]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Preenchimento da tabela (Bottom-Up)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if texto1[i - 1] == texto2[j - 1]:
                # Caracteres iguais: pega diagonal + 1
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # Diferentes: carrega o melhor de cima ou da esquerda
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstrução da string (Backtracking)
    resultado = []
    i, j = m, n
    while i > 0 and j > 0:
        if texto1[i - 1] == texto2[j - 1]:
            resultado.append(texto1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1  # Veio de cima
        else:
            j -= 1  # Veio da esquerda

    return "".join(reversed(resultado))


# ==============================================================================
# 4. DIJKSTRA (Caminho Mínimo em Grafo com Pesos Positivos)
# ==============================================================================

def dijkstra(grafo: Dict[str, Dict[str, int]], inicio: str) -> Dict[str, float]:
    """
    Algoritmo de Dijkstra.
    Entrada: Grafo no formato {'A': {'B': 1, 'C': 4}, ...}
    Retorna: Dicionário com distâncias mínimas {'No': distancia}
    """
    # 1. Inicialização
    distancias = {no: float('inf') for no in grafo}
    distancias[inicio] = 0
    nos_nao_visitados = set(grafo.keys())

    # 2. Loop principal enquanto houver nós a visitar
    while nos_nao_visitados:
        # 3. Encontra o nó não visitado com a menor distância
        u = None
        dist_minima = float('inf')
        for no_candidato in nos_nao_visitados:
            if distancias[no_candidato] < dist_minima:
                dist_minima = distancias[no_candidato]
                u = no_candidato

        # Se o nó com menor distância for infinito, os restantes são inalcançáveis
        if u is None or distancias[u] == float('inf'):
            break

        # 4. Marca o nó como visitado
        nos_nao_visitados.remove(u)

        # 5. Relaxamento das arestas para cada vizinho
        # O grafo pode não ter todos os nós como chaves (se um nó só aparece como destino)
        if u not in grafo: continue
        
        for vizinho, peso in grafo[u].items():
            nova_dist = distancias[u] + peso
            if nova_dist < distancias[vizinho]:
                distancias[vizinho] = nova_dist
                
    return distancias


# ==============================================================================
# EXECUÇÃO DE EXEMPLOS
# ==============================================================================

if __name__ == "__main__":
    print("=== 1. TESTE: FESTA DO STUART ===")
    # Construindo a árvore:
    #       Chefe (10)
    #      /          \
    #    Ger1 (1)     Ger2 (10)
    #      |            |
    #   Dev1 (5)      Dev2 (20)
    dev1 = Funcionario("Dev1", 5)
    dev2 = Funcionario("Dev2", 20)
    dev3 = Funcionario("Dev2", 35)
    ger1 = Funcionario("Gerente1", 1, [dev1, dev3])
    ger2 = Funcionario("Gerente2", 10, [dev2])
    ceo = Funcionario("CEO", 10, [ger1, ger2])

    com, sem = resolver_festa_stuart(ceo)
    print(f"Melhor convivência total: {max(com, sem)}")
    print(f"(Com CEO: {com}, Sem CEO: {sem})")
    print("-" * 40)

    print("=== 2. TESTE: PARENTESES MATRIZES ===")
    # Ex: A(10x30), B(30x5), C(5x60)
    d = [10, 30, 5, 60]
    m_table, s_table = matrix_chain_order(d)
    custo_min = m_table[1][len(d) - 1]
    ordem = reconstruir_parenteses(s_table, 1, len(d) - 1)
    print(f"Custo Mínimo de Multiplicações: {custo_min}")
    print(f"Melhor ordem: {ordem}")
    print("-" * 40)

    print("=== 3. TESTE: LCS ===")
    s1 = "ALGORITMO"
    s2 = "ALIGATOR"
    lcs_str = lcs(s1, s2)
    print(f"Texto 1: {s1}")
    print(f"Texto 2: {s2}")
    print(f"LCS: {lcs_str} (Tamanho: {len(lcs_str)})")
    print("-" * 40)

    print("=== 4. TESTE: DIJKSTRA ===")
    grafo_exemplo = {
        'A': {'B': 10, 'C': 3},
        'B': {'C': 1, 'D': 2},
        'C': {'B': 4, 'D': 8, 'E': 2},
        'D': {'E': 7},
        'E': {'D': 9}
    }
    # Nota: D e E formam um ciclo entre si, mas pesos positivos são OK
    dists = dijkstra(grafo_exemplo, 'A')
    print(f"Distâncias a partir de A: {dists}")