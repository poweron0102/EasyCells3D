from fractions import Fraction
from math import inf


class Simplex:
    def __init__(self, objective_function, constraints, constraint_types, mode='max'):
        """
        Inicializa o solver Simplex.

        Args:
            objective_function: Uma lista de coeficientes da função objetivo.
            constraints: Uma lista de listas, onde cada lista interna representa uma restrição [c1, ..., cn, b].
            constraint_types: Uma lista de strings, uma para cada restrição: '<=', '>=', ou '='.
            mode: 'max' para maximização (padrão) ou 'min' para minimização.
        """
        print("Iniciando a configuração do problema Simplex.")
        self.num_vars = len(objective_function)
        self.mode = mode
        self.constraints = constraints
        self.constraint_types = constraint_types

        # Para minimização, maximizamos o negativo da função objetivo.
        self.original_objective_function = objective_function
        self.objective_function = objective_function if mode == 'max' else [-c for c in objective_function]

        self.tableau = []
        self.solution = {}
        self.artificial_vars_cols = []

        self.needs_two_phase = any(ct in ['>=', '='] for ct in self.constraint_types)

        if self.needs_two_phase:
            print("Detectada a necessidade do Método de Duas Fases (restrições '>=' ou '=').")
            self._initialize_phase1_tableau()
        else:
            print("Utilizando o Método Simplex Padrão (apenas restrições '<=').")
            self._initialize_standard_tableau()

    def _initialize_standard_tableau(self):
        """
        Inicializa o tableau simplex para um problema de maximização padrão (todas as restrições <=).
        """
        print("Inicializando o tableau para o método padrão.")
        num_slack_vars = len(self.constraints)
        num_total_vars = self.num_vars + num_slack_vars

        # Cabeçalho
        header = ['Base'] + [f'x{i + 1}' for i in range(self.num_vars)] + \
                 [f's{i + 1}' for i in range(num_slack_vars)] + ['b']
        self.tableau.append(header)

        # Linhas de restrição
        for i, constraint in enumerate(self.constraints):
            row = [0] * (num_total_vars + 2)
            row[0] = f's{i + 1}'  # Variável básica
            for j in range(self.num_vars):
                row[j + 1] = Fraction(constraint[j])
            row[self.num_vars + i + 1] = Fraction(1)  # Variável de folga
            row[-1] = Fraction(constraint[-1])
            self.tableau.append(row)

        # Linha da função objetivo (linha Z)
        obj_row = [0] * (num_total_vars + 2)
        obj_row[0] = 'Z'
        for i in range(self.num_vars):
            obj_row[i + 1] = -Fraction(self.objective_function[i])
        obj_row[-1] = Fraction(0)
        self.tableau.append(obj_row)

    def _initialize_phase1_tableau(self):
        """
        Inicializa o tableau para a Fase 1 do Método Simplex de Duas Fases.
        """
        print("\n--- FASE 1: Encontrar uma solução básica viável ---")
        print("Objetivo da Fase 1: Minimizar a soma das variáveis artificiais (W).")

        num_slack = self.constraint_types.count('<=')
        num_surplus = self.constraint_types.count('>=')
        num_artificial = self.constraint_types.count('>=') + self.constraint_types.count('=')

        self.artificial_vars_cols = []

        num_total_vars = self.num_vars + num_slack + num_surplus + num_artificial

        # Cabeçalho
        header = ['Base'] + [f'x{i + 1}' for i in range(self.num_vars)]
        s_idx, u_idx, a_idx = 1, 1, 1

        # Gerar nomes de variáveis para o cabeçalho
        slack_vars = []
        surplus_vars = []
        artificial_vars = []
        for ct in self.constraint_types:
            if ct == '<=':
                slack_vars.append(f's{s_idx}')
                s_idx += 1
            elif ct == '>=':
                surplus_vars.append(f'u{u_idx}')
                artificial_vars.append(f'a{a_idx}')
                u_idx += 1
                a_idx += 1
            elif ct == '=':
                artificial_vars.append(f'a{a_idx}')
                a_idx += 1

        header += slack_vars + surplus_vars + artificial_vars + ['b']
        self.tableau.append(header)

        # Linhas de restrição
        s_idx, u_idx, a_idx = 1, 1, 1
        artificial_rows_indices = []
        for i, constraint in enumerate(self.constraints):
            row = [Fraction(0)] * (num_total_vars + 2)

            # Coeficientes de x_i
            for j in range(self.num_vars):
                row[j + 1] = Fraction(constraint[j])

            # Adicionar variáveis de folga, excesso e artificiais
            if self.constraint_types[i] == '<=':
                row[0] = f's{s_idx}'
                row[self.num_vars + s_idx] = Fraction(1)
                s_idx += 1
            elif self.constraint_types[i] == '>=':
                row[0] = f'a{a_idx}'
                # Variável de excesso
                surplus_col = self.num_vars + num_slack + u_idx
                row[surplus_col] = Fraction(-1)
                u_idx += 1
                # Variável artificial
                artificial_col = self.num_vars + num_slack + num_surplus + a_idx
                row[artificial_col] = Fraction(1)
                self.artificial_vars_cols.append(artificial_col)
                artificial_rows_indices.append(len(self.tableau))  # Armazena o índice da linha
                a_idx += 1
            elif self.constraint_types[i] == '=':
                row[0] = f'a{a_idx}'
                # Variável artificial
                artificial_col = self.num_vars + num_slack + num_surplus + a_idx
                row[artificial_col] = Fraction(1)
                self.artificial_vars_cols.append(artificial_col)
                artificial_rows_indices.append(len(self.tableau))  # Armazena o índice da linha
                a_idx += 1

            row[-1] = Fraction(constraint[-1])
            self.tableau.append(row)

        # Fase 1 - Linha da Função Objetivo (linha W)
        # O objetivo é maximizar W' = -soma(a_i), representado no tableau por W' + soma(a_i) = 0.
        # Portanto, a linha W inicial (antes de ser canônica) tem coeficiente 1 para cada variável artificial.
        w_row = [Fraction(0)] * (num_total_vars + 2)
        w_row[0] = 'W'

        # 1. Define o coeficiente 1 para cada variável artificial na linha W (Passo que faltava)
        for col_idx in self.artificial_vars_cols:
            w_row[col_idx] = Fraction(1)

        # 2. Agora, tornamos a linha W canônica. Como as variáveis artificiais estão na base,
        #    seus coeficientes na linha W devem ser zero. Para isso, subtraímos de W
        #    cada linha que tem uma variável artificial como variável básica.
        for r_idx in artificial_rows_indices:
            art_row = self.tableau[r_idx]
            # A operação zera o coeficiente da variável artificial 'a_i' na linha W.
            for j in range(1, len(w_row)):
                w_row[j] -= art_row[j]

        self.tableau.append(w_row)
        print("Tableau da Fase 1 pronto para iniciar as iterações:")

    def _prepare_for_phase2(self):
        """
        Prepara o tableau para a Fase 2 após uma Fase 1 bem-sucedida.
        """
        print("\n--- FASE 2: Encontrar a solução ótima do problema original ---")
        print("Fase 1 bem-sucedida. Removendo variáveis artificiais e restaurando a função objetivo original (Z).")

        # Remove a linha W
        self.tableau.pop()

        # Cria e adiciona a linha Z original
        obj_row = [Fraction(0)] * (len(self.tableau[0]))
        obj_row[0] = 'Z'
        for i in range(self.num_vars):
            obj_row[i + 1] = -Fraction(self.objective_function[i])
        self.tableau.append(obj_row)

        print("Tableau com a função objetivo Z (antes de canonizar):")
        print(self)

        # Torna a linha Z canônica para a base atual
        print("Ajustando a linha Z para a forma canônica com a base atual...")
        z_row = self.tableau[-1]
        for i in range(1, len(self.tableau) - 1):  # Para cada linha de variável básica
            basic_var = self.tableau[i][0]
            try:
                # Encontra a coluna desta variável básica
                col_idx = self.tableau[0].index(basic_var)
                if z_row[col_idx] != 0:
                    print(f"A variável básica '{basic_var}' tem um coeficiente não-nulo na linha Z. Zerando...")
                    factor = z_row[col_idx]
                    pivot_row = self.tableau[i]
                    # z_row_new = z_row_old - factor * pivot_row
                    for j in range(1, len(z_row)):
                        z_row[j] -= factor * pivot_row[j]
            except ValueError:
                # Pode acontecer se uma variável básica for artificial e estiver prestes a ser removida
                pass

        # Remove as colunas das variáveis artificiais
        # Fazemos isso por último para não bagunçar os índices das colunas durante a canonização da linha Z
        print("Removendo colunas das variáveis artificiais...")
        self.artificial_vars_cols.sort(reverse=True)  # Ordena decrescente para não bagunçar os índices
        for col_idx in self.artificial_vars_cols:
            for row in self.tableau:
                del row[col_idx]
        self.artificial_vars_cols = []

        print("Tableau da Fase 2 pronto para iniciar as iterações:")

    @staticmethod
    def _format_value(value):
        """ Formata um valor para exibição no tableau. """
        if isinstance(value, Fraction):
            # Tenta exibir como inteiro se possível
            if value.denominator == 1:
                return str(value.numerator)
            s_value = str(value)
            if len(s_value) > 8:
                return f"{float(value):.2f}"
            return s_value
        return str(value)

    def __str__(self):
        """
        Retorna uma representação em string formatada do tableau.
        """
        if not self.tableau:
            return ""

        display_tableau = [[self._format_value(item) for item in row] for row in self.tableau]
        col_widths = [max(len(item) for item in col) for col in zip(*display_tableau)]

        header = " | ".join(f"{item:<{col_widths[i]}}" for i, item in enumerate(display_tableau[0]))
        separator = "-".join("-" * (width + 2) for width in col_widths)

        rows = []
        for row in display_tableau[1:]:
            base_var = f"{row[0]:<{col_widths[0]}}"
            values = " | ".join(f"{val:>{col_widths[j + 1]}}" for j, val in enumerate(row[1:]))
            rows.append(f"{base_var} | {values}")

        return f"{header}\n{separator}\n" + "\n".join(rows)

    def _find_pivot_column(self):
        z_row = self.tableau[-1]
        # Ignora a coluna 'b'
        relevant_part = z_row[1:-1]
        if not relevant_part:
            return -1
        most_negative = min(relevant_part, default=0)
        return z_row.index(most_negative) if most_negative < 0 else -1

    def _find_pivot_row(self, pivot_col_idx):
        min_ratio = inf
        pivot_row_idx = -1
        for i in range(1, len(self.tableau) - 1):
            # Verifica se o elemento pivô é positivo
            if self.tableau[i][pivot_col_idx] > 0:
                ratio = self.tableau[i][-1] / self.tableau[i][pivot_col_idx]
                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row_idx = i
        return pivot_row_idx

    def _pivot(self, pivot_row_idx, pivot_col_idx):
        entering_variable = self.tableau[0][pivot_col_idx]

        # Normaliza a linha do pivô
        pivot_row = self.tableau[pivot_row_idx]
        pivot_val = pivot_row[pivot_col_idx]
        for j in range(1, len(pivot_row)):
            pivot_row[j] /= pivot_val
        pivot_row[0] = entering_variable

        # Atualiza as outras linhas
        for i in range(1, len(self.tableau)):
            if i != pivot_row_idx:
                row = self.tableau[i]
                factor = row[pivot_col_idx]
                for j in range(1, len(row)):
                    row[j] -= factor * pivot_row[j]

    def _extract_solution(self):
        z_row = self.tableau[-1]
        final_z = z_row[-1]
        self.solution['Z'] = -final_z if self.mode == 'min' else final_z

        for i in range(1, len(self.tableau) - 1):
            base_var = self.tableau[i][0]
            self.solution[base_var] = self.tableau[i][-1]

        for i in range(1, self.num_vars + 1):
            var = f'x{i}'
            if var not in self.solution:
                self.solution[var] = Fraction(0)

    def _run_simplex_iterations(self, phase_name):
        iteration = 1
        while True:
            print(f"\n--- {phase_name} - Iteração {iteration} ---")
            print(self)

            pivot_col_idx = self._find_pivot_column()
            if pivot_col_idx == -1:
                print(
                    f"\n>> Condição de parada da {phase_name} alcançada (não há mais valores negativos na linha de objetivo).")
                break

            entering_variable = self.tableau[0][pivot_col_idx]
            print(
                f"\nDecisão: A variável {entering_variable} entra na base (coluna com o valor mais negativo na linha de objetivo).")

            pivot_row_idx = self._find_pivot_row(pivot_col_idx)
            if pivot_row_idx == -1:
                print("Problema ilimitado.")
                return False  # Indica falha

            leaving_variable = self.tableau[pivot_row_idx][0]
            pivot_element = self.tableau[pivot_row_idx][pivot_col_idx]
            print(f"Decisão: A variável {leaving_variable} sai da base (menor razão não-negativa).")
            print(
                f"Elemento Pivô: {self._format_value(pivot_element)} (na linha {leaving_variable} e coluna {entering_variable})")

            self._pivot(pivot_row_idx, pivot_col_idx)
            iteration += 1
        return True  # Indica sucesso

    def solve(self):
        if self.needs_two_phase:
            # --- Fase 1 ---
            if not self._run_simplex_iterations("Fase 1"):
                self.solution = {"Status": "Ilimitado"}
                return  # Problema é ilimitado

            w_value = self.tableau[-1][-1]
            print(f"\nFim da Fase 1. Valor final de W = {self._format_value(w_value)}")

            # Se W não for zero, o problema é inviável
            # Usa uma pequena tolerância para comparações de ponto flutuante
            if abs(w_value) > 1e-9:
                print(">> Problema inviável: não foi possível zerar a função objetivo da Fase 1 (W).")
                self.solution = {"Status": "Inviável"}
                return

            # Verifica se alguma variável artificial ainda está na base com valor não-nulo.
            # Pode acontecer em casos degenerados, mas se o valor for 0, está tudo bem.
            for i in range(1, len(self.tableau) - 1):
                if self.tableau[i][0].startswith('a') and abs(self.tableau[i][-1]) > 1e-9:
                    print(">> Problema inviável: variável artificial permanece na base com valor não-nulo.")
                    self.solution = {"Status": "Inviável"}
                    return

            # --- Fase 2 ---
            self._prepare_for_phase2()
            if not self._run_simplex_iterations("Fase 2"):
                self.solution = {"Status": "Ilimitado"}
                return  # Problema é ilimitado

        else:
            # --- Simplex Padrão ---
            print("\nIniciando o método Simplex Padrão...")
            if not self._run_simplex_iterations("Simplex Padrão"):
                self.solution = {"Status": "Ilimitado"}
                return  # Problema é ilimitado

        print("\n>> Solução ótima encontrada.")
        print(self)
        self._extract_solution()
        print("\nSolução Final:")
        for var, value in sorted(self.solution.items()):
            if var.startswith('x') or var == 'Z':
                print(f"{var} = {self._format_value(value)}")


if __name__ == '__main__':
    objective = [6, 4, 2, 8, 8]

    constraints = [
    [1, 0, 0,    0,    0,  10],  # X1 >= 10
    [1, 1, 0,    0,    0,  60],  # X1 + X2 >= 60
    [1, 1, 1,    0,    0, 110],  # Soma >= 110
    [1, 0, 0,    0,    0, 100],  # X1 <= 100
    [0, 1, 0, -100,    0,   0],  # X2 - 100f2 <= 0
    [0, 0, 1,    0, -100,   0],  # X3 - 100f3 <= 0
    [0, 0, 0,    1,    0,   0],
    [0, 0, 0,    0,    1,   0],
]

    constraint_types_min = [">=", ">=", ">=", "<=", "<=", "<=", "=", "="]

    simplex_min = Simplex(objective, constraints, constraint_types_min, mode='min')
    simplex_min.solve()