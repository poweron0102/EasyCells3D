import numpy as np
from scipy.optimize import linprog


def resolver_otimizacao(nome_produto, obj_coeffs, restricoes_raw, tipos_restricoes):
    """
    Resolve o problema de otimização linear mista.

    Args:
        nome_produto: String com nome (A ou B)
        obj_coeffs: Vetor da função objetivo (c)
        restricoes_raw: Lista de listas conforme suas anotações [coefs..., rhs]
        tipos_restricoes: Lista de strings '>=' ou '<='
    """

    print(f"\n{'=' * 10} RESOLVENDO PRODUTO {nome_produto} {'=' * 10}")

    # 1. Separar Matriz A (coeficientes) e Vetor b (lado direito)
    # O linprog trabalha nativamente com Ax <= b.
    # Se a restrição for >=, multiplicamos tudo por -1 para inverter.

    A_ub = []  # Upper bound matrix (Inequalities)
    b_ub = []  # Upper bound vector (RHS)

    for i, row in enumerate(restricoes_raw):
        coeffs = row[:-1]  # Todos menos o último
        rhs = row[-1]  # O último elemento é o valor
        tipo = tipos_restricoes[i]

        if tipo == ">=":
            # Inverte sinal para transformar >= em <=
            A_ub.append([-c for c in coeffs])
            b_ub.append(-rhs)
        else:
            # Mantém como está
            A_ub.append(coeffs)
            b_ub.append(rhs)

    # 2. Definir limites das variáveis (Bounds)
    # Variáveis X1, X2, X3: 0 a infinito
    # Variáveis f2, f3: 0 a 1 (Binárias)
    bounds = [
        (0, None),  # X1
        (0, None),  # X2
        (0, None),  # X3
        (0, 1),  # f2 (Binário)
        (0, 1)  # f3 (Binário)
    ]

    # 3. Definir quais variáveis são inteiras
    # 0 = contínua, 1 = inteira. As duas últimas (f2, f3) são inteiras.
    integrality = [0, 0, 0, 1, 1]

    # 4. Resolver
    res = linprog(c=obj_coeffs, A_ub=A_ub, b_ub=b_ub, bounds=bounds, integrality=integrality, method='highs')

    # 5. Mostrar Resultados
    if res.success:
        print(f"Status: Otimização Encontrada!")
        print(f"Custo Mínimo (Z): {res.fun:.2f}")
        print("-" * 30)
        print(f"Produção Dia 1 (X1): {res.x[0]:.2f}")
        print(f"Produção Dia 2 (X2): {res.x[1]:.2f}")
        print(f"Produção Dia 3 (X3): {res.x[2]:.2f}")
        print("-" * 30)
        # Arredondando f para garantir visualização limpa (0 ou 1)
        print(f"Ligar Máquina Dia 2 (f2): {int(round(res.x[3]))}")
        print(f"Ligar Máquina Dia 3 (f3): {int(round(res.x[4]))}")
    else:
        print("Não foi possível encontrar uma solução ótima.")
        print(res.message)


# ==========================================
# DADOS DO PRODUTO A
# ==========================================
# Variáveis na ordem: [X1, X2, X3, f2, f3]
obj_A = [3, 2, 1, 5, 5]

# Matriz das suas anotações (Coeficientes + RHS no final)
# Nota: Removi as restrições f <= 1 da matriz pois tratei no 'bounds' do solver,
# mas mantive a lógica das suas equações principais.
restr_A = [
    [1, 0, 0, 0, 0, 100],  # X1 >= 100
    [1, 1, 0, 0, 0, 300],  # X1 + X2 >= 300
    [1, 1, 1, 0, 0, 600],  # Soma >= 600
    [1, 0, 0, 0, 0, 300],  # X1 <= 300
    [0, 1, 0, -300, 0, 0],  # X2 - 300f2 <= 0
    [0, 0, 1, 0, -300, 0]  # X3 - 300f3 <= 0
]
# Tipos correspondentes a cada linha acima
tipos_A = [">=", ">=", ">=", "<=", "<=", "<="]

resolver_otimizacao("A", obj_A, restr_A, tipos_A)

# ==========================================
# DADOS DO PRODUTO B
# ==========================================
obj_B = [6, 4, 2, 8, 8]

restr_B = [
    [1, 0, 0, 0, 0, 10],  # X1 >= 10
    [1, 1, 0, 0, 0, 60],  # X1 + X2 >= 60
    [1, 1, 1, 0, 0, 110],  # Soma >= 110
    [1, 0, 0, 0, 0, 100],  # X1 <= 100
    [0, 1, 0, -100, 0, 0],  # X2 - 100f2 <= 0
    [0, 0, 1, 0, -100, 0]  # X3 - 100f3 <= 0
]
tipos_B = [">=", ">=", ">=", "<=", "<=", "<="]

resolver_otimizacao("B", obj_B, restr_B, tipos_B)