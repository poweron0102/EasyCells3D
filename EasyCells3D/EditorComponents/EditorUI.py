import math

import pyray as rl

from EasyCells3D import Game
from EasyCells3D.Components import Component, Item
from EasyCells3D.Components.CameraUI import RenderableUI
from EasyCells3D.Geometry import Vec3


# Componente de teste para demonstrar a adição dinâmica
class DummyComponent(Component):
    def init(self):
        self.speed = 400.0
        self.jump_force = 200.0
        self.name = "Player Dummy"


class EditorUI(RenderableUI):

    def __init__(self, running_game: Game):
        self.running_game = running_game

    def init(self):
        super().init()
        self.selected_item: Item | None = None
        self.camera_mode_2d = True  # Estado para o botão 2D/3D

        # Variáveis de Scroll
        self.cam_scroll_y = 0.0
        self.cam_content_h = 0.0

        self.hier_scroll_y = 0.0
        self.hier_content_h = 0.0

        self.comp_scroll_y = 0.0
        self.comp_content_h = 0.0

        # Variáveis de layout e controle
        rl.gui_set_style(rl.DEFAULT, rl.TEXT_SIZE, 16)

    def _update_scroll(self, panel_rect: rl.Rectangle, current_scroll: float, content_h: float) -> float:
        """Atualiza o valor de scroll com base na roda do rato, limitando aos limites do conteúdo."""
        mouse_pos = rl.get_mouse_position()

        # Só dá scroll se o rato estiver por cima do painel
        if rl.check_collision_point_rec(mouse_pos, panel_rect):
            wheel = rl.get_mouse_wheel_move()
            current_scroll += wheel * 30.0  # 30 pixels por movimento da roda

        panel_inner_h = panel_rect.height - 30  # Desconta a altura do cabeçalho do grupo

        # Limita o máximo que pode "subir" o conteúdo
        max_scroll_up = min(0.0, panel_inner_h - content_h)

        # O scroll atual nunca deve ser > 0 e nunca menor que max_scroll_up
        return max(max_scroll_up, min(0.0, current_scroll))

    def _draw_vector3_input(self, x, y, w, label, vec3_obj):
        """Desenha os valores de um Vec3 responsivo à largura disponível (w)."""
        rl.gui_label(rl.Rectangle(x, y, 60, 20), label)

        # Espaço restante dividido para X, Y, Z
        rem_w = w - 65
        part_w = rem_w / 3

        # X
        rl.gui_label(rl.Rectangle(x + 65, y, 15, 20), "X:")
        rl.gui_text_box(rl.Rectangle(x + 65 + 15, y, part_w - 20, 20), f"{vec3_obj.x:.1f}", 10, False)

        # Y
        rl.gui_label(rl.Rectangle(x + 65 + part_w, y, 15, 20), "Y:")
        rl.gui_text_box(rl.Rectangle(x + 65 + part_w + 15, y, part_w - 20, 20), f"{vec3_obj.y:.1f}", 10, False)

        # Z
        rl.gui_label(rl.Rectangle(x + 65 + part_w * 2, y, 15, 20), "Z:")
        rl.gui_text_box(rl.Rectangle(x + 65 + part_w * 2 + 15, y, part_w - 20, 20), f"{vec3_obj.z:.1f}", 10,
                        False)

    def _draw_hierarchy_node(self, item: Item, x, y, w, indent) -> float:
        """Função recursiva para desenhar a árvore de itens dinamicamente ajustada à largura."""
        name = getattr(item, 'name', f"Item_{id(item) % 10000}")

        is_selected = (item == self.selected_item)

        # Desenha a linhazinha da árvore (|- ) se houver indentação
        if indent > 0:
            rl.draw_line(int(x + indent - 10), int(y + 10), int(x + indent - 2), int(y + 10), rl.GRAY)
            rl.draw_line(int(x + indent - 10), int(y - 15), int(x + indent - 10), int(y + 10), rl.GRAY)

        rect = rl.Rectangle(x + indent, y, w - indent - 10, 20)

        # Botão para seleção
        if rl.gui_button(rect, name):
            self.selected_item = item

        y += 25
        # Chama a função para todos os filhos dinamicamente
        for child in item.children:
            y = self._draw_hierarchy_node(child, x, y, w, indent + 15)

        return y

    def render(self):
        # ==========================================
        # CÁLCULO DE LAYOUT DINÂMICO (Responsividade)
        # ==========================================
        sw = rl.get_screen_width()
        sh = rl.get_screen_height()
        pad = 10  # Espaçamento padrão entre painéis

        # 1. Canvas ocupa 60% da tela na esquerda
        canvas_w_pct = 0.60
        canvas_x = pad
        canvas_y = pad
        canvas_w = (sw * canvas_w_pct) - (pad * 1.5)
        canvas_h = sh - (pad * 2)

        # 2. Sidebar (Painel Direito) ocupa o resto
        sidebar_x = canvas_x + canvas_w + pad
        sidebar_y = pad
        sidebar_w = sw - canvas_w - (pad * 3)
        sidebar_h = sh - (pad * 2)

        # 3. Divisões da Sidebar (conforme Mockup)
        top_h_pct = 0.20  # Câmera e Transform ocupam 20% da altura do painel direito
        sidebar_left_w_pct = 0.35  # Câmera e Hierarquia ocupam 35% da largura do painel direito

        # -- Topo Esquerda (Câmera Configs) --
        cam_x = sidebar_x
        cam_y = sidebar_y
        cam_w = sidebar_w * sidebar_left_w_pct - (pad * 0.5)
        cam_h = sidebar_h * top_h_pct

        # -- Topo Direita (Transform) --
        tf_x = cam_x + cam_w + pad
        tf_y = sidebar_y
        tf_w = sidebar_w - cam_w - pad
        tf_h = cam_h

        # -- Base Esquerda (Hierarquia) --
        hier_x = sidebar_x
        hier_y = cam_y + cam_h + pad
        hier_w = cam_w
        hier_h = sidebar_h - cam_h - pad

        # -- Base Direita (Componentes) --
        comp_x = tf_x
        comp_y = hier_y
        comp_w = tf_w
        comp_h = hier_h

        # ==========================================
        # DESENHO DA INTERFACE
        # ==========================================

        # 1. GAME CANVAS
        canvas_rect = rl.Rectangle(canvas_x, canvas_y, canvas_w, canvas_h)
        rl.draw_rectangle_lines_ex(canvas_rect, 2, rl.BLACK)
        text_size = int(canvas_w * 0.05)  # Texto escala com o canvas
        rl.draw_text("Game canvas", int(canvas_x + canvas_w / 2 - text_size * 3), int(canvas_y + canvas_h / 2),
                     text_size, rl.LIGHTGRAY)

        # 2. CÂMERA CONFIGS
        cam_rect = rl.Rectangle(cam_x, cam_y, cam_w, cam_h)
        rl.gui_group_box(cam_rect, "Camera Configs")

        self.cam_scroll_y = self._update_scroll(cam_rect, self.cam_scroll_y, self.cam_content_h)

        rl.begin_scissor_mode(int(cam_x), int(cam_y + 10), int(cam_w), int(cam_h - 10))

        start_y = cam_y + 20 + self.cam_scroll_y

        # Controles de Playback (Play, Pause, Stop)
        control_h = 40
        btn_w3 = (cam_w - 30) / 3  # Divide o espaço para 3 botões com 5px de espaçamento

        if rl.gui_button(rl.Rectangle(cam_x + 10, start_y, btn_w3, 30), "Play"):
            pass  # Implementar lógica de Play aqui

        if rl.gui_button(rl.Rectangle(cam_x + 10 + btn_w3 + 5, start_y, btn_w3, 30), "Pause"):
            pass  # Implementar lógica de Pause aqui

        if rl.gui_button(rl.Rectangle(cam_x + 10 + (btn_w3 + 5) * 2, start_y, btn_w3, 30), "Stop"):
            pass  # Implementar lógica de Stop aqui

        # Botões 2D / 3D
        btn_y = start_y + control_h + 10
        btn_w = (cam_w - 30) / 2
        if rl.gui_button(rl.Rectangle(cam_x + 10, btn_y, btn_w, 25), "2D"):
            self.camera_mode_2d = True
        if rl.gui_button(rl.Rectangle(cam_x + 10 + btn_w + 10, btn_y, btn_w, 25), "3D"):
            self.camera_mode_2d = False

        self.cam_content_h = (btn_y + 25) - (cam_y + 20 + self.cam_scroll_y)

        rl.end_scissor_mode()

        # 3. TRANSFORM
        tf_rect = rl.Rectangle(tf_x, tf_y, tf_w, tf_h)
        rl.gui_group_box(tf_rect, "Transform")

        if self.selected_item:
            tf = self.selected_item.transform

            # Espaçamento dinâmico para os campos
            spacing_y = (tf_h - 20) / 4

            self._draw_vector3_input(tf_x + 10, tf_y + spacing_y, tf_w - 20, "Position", tf.position)

            euler = tf.rotation.to_euler_angles()
            rot_vec = Vec3(math.degrees(euler.x), math.degrees(euler.y), math.degrees(euler.z))
            self._draw_vector3_input(tf_x + 10, tf_y + spacing_y * 2, tf_w - 20, "Rotation", rot_vec)

            self._draw_vector3_input(tf_x + 10, tf_y + spacing_y * 3, tf_w - 20, "Scale", tf.scale)

        # 4. HIERARQUIA
        hier_rect = rl.Rectangle(hier_x, hier_y, hier_w, hier_h)
        rl.gui_group_box(hier_rect, "Hierarquia")

        self.hier_scroll_y = self._update_scroll(hier_rect, self.hier_scroll_y, self.hier_content_h)

        # Inicia corte para evitar que itens da hierarquia vazem da caixa
        rl.begin_scissor_mode(int(hier_x), int(hier_y + 10), int(hier_w), int(hier_h - 10))

        y_offset = hier_y + 20 + self.hier_scroll_y

        # Botões do topo
        btn_add_w = hier_w - 20
        if rl.gui_button(rl.Rectangle(hier_x + 10, y_offset, btn_add_w, 20), "+ Add Root"):
            new_item = self.running_game.CreateItem()
            new_item.name = f"Item_{len(self.running_game.item_list)}"
        y_offset += 25

        if self.selected_item and rl.gui_button(rl.Rectangle(hier_x + 10, y_offset, btn_add_w, 20), "+ Add Child"):
            new_child = self.selected_item.CreateChild()
            new_child.name = f"Filho_{len(self.selected_item.children)}"
        y_offset += 35

        # Percorre a lista de itens
        for item in self.running_game.item_list:
            y_offset = self._draw_hierarchy_node(item, hier_x + 10, y_offset, hier_w, 0)

        self.hier_content_h = y_offset - (hier_y + 20 + self.hier_scroll_y)

        rl.end_scissor_mode()

        # 5. COMPONENTES
        comp_rect = rl.Rectangle(comp_x, comp_y, comp_w, comp_h)
        rl.gui_group_box(comp_rect, "Componentes")

        self.comp_scroll_y = self._update_scroll(comp_rect, self.comp_scroll_y, self.comp_content_h)

        rl.begin_scissor_mode(int(comp_x), int(comp_y + 10), int(comp_w), int(comp_h - 10))

        y_comp = comp_y + 20 + self.comp_scroll_y

        if self.selected_item:

            # --- CAMPO PARA RENOMEAR O ITEM ---
            rl.gui_label(rl.Rectangle(comp_x + 15, y_comp, 50, 20), "Nome:")
            item_name = getattr(self.selected_item, 'name', f"Item_{id(self.selected_item) % 10000}")
            rl.gui_text_box(rl.Rectangle(comp_x + 65, y_comp, comp_w - 85, 20), item_name, 50, False)
            y_comp += 30
            # ----------------------------------

            if rl.gui_button(rl.Rectangle(comp_x + 10, y_comp, comp_w - 20, 25), "+ Add Dummy Component"):
                self.selected_item.AddComponent(DummyComponent())
            y_comp += 35

            for comp_class, comp_instance in self.selected_item.components.items():
                if comp_class == Component:
                    continue

                # Caixa do Componente (Título)
                rl.gui_label(rl.Rectangle(comp_x + 10, y_comp, comp_w - 20, 20), f"[{comp_class.__name__}]")
                rl.draw_line(int(comp_x + 10), int(y_comp + 20), int(comp_x + comp_w - 10), int(y_comp + 20), rl.GRAY)
                y_comp += 25

                # Propriedades
                label_w = comp_w * 0.4
                input_w = comp_w * 0.5

                for attr_name, attr_value in vars(comp_instance).items():
                    if attr_name in ['item', 'game']:
                        continue

                    # Responsivo: Label fica na esquerda, TextBox na direita do painel
                    rl.gui_label(rl.Rectangle(comp_x + 15, y_comp, label_w, 20), f"{attr_name}:")
                    val_str = str(attr_value)
                    rl.gui_text_box(rl.Rectangle(comp_x + 15 + label_w, y_comp, input_w, 20), val_str, 50,
                                    False)
                    y_comp += 25

                y_comp += 10  # Espaçamento extra

        self.comp_content_h = y_comp - (comp_y + 20 + self.comp_scroll_y)

        rl.end_scissor_mode()