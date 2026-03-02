import raylibpy as rl

from EasyCells3D import Game
from EasyCells3D.Components import Item, Animation2D, Sprite, Animator2D, Component
from EasyCells3D.PhysicsComponents import Rigidbody, Collider

frame_size = 32
animations_info = [
    ("Idle", "Idle (32x32).png"),
    ("Run", "Run (32x32).png"),
    ("Jump", "Jump (32x32).png"),
    ("Double Jump", "Double Jump (32x32).png"),
    ("Fall", "Fall (32x32).png"),
    ("Wall Jump", "Wall Jump (32x32).png"),
    ("Hit", "Hit (32x32).png")
]


def load_player(game: 'Game', folder_name: str) -> Item:
    """
    Carrega as sprites do jogador de uma pasta, unifica em uma única textura
    separada por index_y e cria o Item com Sprite e Animator2D.
    """

    images = []
    max_width = 0

    #Carregar todas as imagens para a memória (RAM) e descobrir a largura máxima
    for anim_name, file_name in animations_info:
        # A classe Sprite usa "Assets/{path}", então seguimos a mesma lógica
        img_path = f"Assets/{folder_name}/{file_name}"
        img = rl.load_image(img_path)
        images.append((anim_name, img))

        if img.width > max_width:
            max_width = img.width

    #Criar uma imagem em branco grande o suficiente para conter todas as animações
    total_height = len(animations_info) * frame_size
    atlas_image = rl.gen_image_color(max_width, total_height, rl.BLANK)

    dict_animations: dict[str, Animation2D] = {}

    #Desenhar cada spritesheet na imagem principal (atlas) e configurar as animações
    for index_y, (anim_name, img) in enumerate(images):
        src_rec = rl.Rectangle(0, 0, img.width, img.height)
        dst_rec = rl.Rectangle(0, index_y * frame_size, img.width, img.height)

        # Copia os pixels da imagem individual para o atlas
        rl.image_draw(atlas_image, img, src_rec, dst_rec, rl.WHITE)
        num_frames = int(img.width // frame_size)

        # Hit deve voltar pro Idle.
        on_end = "Idle" if anim_name == "Hit" else None

        # Ajuste de velocidade (Run e Hit costumam ser mais rápidos que Idle)
        speed = 0.05 if anim_name in ["Run", "Hit"] else 0.1

        # Cria o objeto de animação e adiciona ao dicionário
        dict_animations[anim_name] = Animation2D(
            speed=speed,
            frames=list(range(num_frames)),
            index_y=index_y,
            on_end=on_end
        )

        # Libera a imagem individual da memória, pois já foi copiada para o atlas
        rl.unload_image(img)

    final_texture = rl.load_texture_from_image(atlas_image)
    rl.unload_image(atlas_image)

    #Criar o Item e adicionar os Componentes
    player_item = game.CreateItem()
    player_item.AddComponent(Sprite(final_texture, size=(frame_size, frame_size)))
    player_item.AddComponent(Animator2D(dict_animations, "Idle"))

    return player_item



class Player(Component):
    def init(self):
        # Pegando as referências dos componentes necessários no mesmo Item
        self.rb: Rigidbody = self.GetComponent(Rigidbody)
        self.animator: Animator2D = self.GetComponent(Animator2D)
        self.sprite: Sprite = self.GetComponent(Sprite)
        self.collider: Collider = self.GetComponent(Collider)

        # Variáveis de movimento
        self.move_speed = 300.0
        self.jump_force = 450.0

        # Variáveis de estado
        self.can_double_jump = False
        self.is_hit = False
        self.is_grounded = False
        self.is_touching_wall = False
        self.wall_direction = 0

    def loop(self):
        # Se estiver na animação de Hit, bloqueia outras ações de movimento
        if self.is_hit:
            if self.animator.current_animation != "Hit":
                self.is_hit = False
            else:
                return

        self._check_environment()
        self._handle_movement()
        self._update_animations()

    def take_hit(self):
        """Função chamada quando o player toma dano."""
        self.is_hit = True
        self.animator.current_animation = "Hit"

    def _check_environment(self):
        """
        Verifica se o player está no chão ou na parede.
        Para um jogo real, você usaria ray_cast do seu Collider aqui.
        """
        # Exemplo simplificado usando a velocidade do Rigidbody para o chão
        # (O ideal seria usar Collider.ray_cast_static apontando para baixo)
        self.is_grounded = abs(self.rb.velocity.y) < 1.0

        if self.is_grounded:
            self.can_double_jump = True

        # Stub para parede: Você implementaria um ray_cast lateral aqui.
        self.is_touching_wall = False
        self.wall_direction = 0

    def _handle_movement(self):
        # Input horizontal
        move_input = 0
        if rl.is_key_down(rl.KEY_RIGHT): move_input += 1
        if rl.is_key_down(rl.KEY_LEFT): move_input  -= 1

        # Virar o sprite baseando-se no input
        if move_input > 0:
            self.sprite.horizontal_flip = False
        elif move_input < 0:
            self.sprite.horizontal_flip = True

        # Aplicar velocidade horizontal
        self.rb.velocity.x = move_input * self.move_speed

        # Lógica de pulo
        if rl.is_key_pressed(rl.KEY_SPACE):
            if self.is_grounded:
                self.rb.velocity.y = -self.jump_force
            elif self.is_touching_wall and move_input == self.wall_direction:
                # Wall Jump: empurra pra cima e pro lado oposto da parede
                self.rb.velocity.y = -self.jump_force
                self.rb.velocity.x = -self.wall_direction * self.move_speed
                self.animator.current_animation = "Jump"
            elif self.can_double_jump:
                self.rb.velocity.y = -self.jump_force
                self.can_double_jump = False
                self.animator.current_animation = "Double Jump"

    def _update_animations(self):
        # Não sobrescrever as animações de hit ou de double jump se elas ainda devem tocar
        if self.animator.current_animation in ["Hit", "Double Jump"] and not self.is_grounded:
            # Vamos deixar a animação de Double Jump tocar até começar a cair, por exemplo.
            if self.animator.current_animation == "Double Jump" and self.rb.velocity.y > 0:
                pass  # Continua para a lógica de queda abaixo
            else:
                return

        move_input = 0
        if rl.is_key_down(rl.KEY_RIGHT): move_input += 1
        if rl.is_key_down(rl.KEY_LEFT): move_input -= 1

        # Lógica de Máquina de Estados de Animação
        if self.is_touching_wall and not self.is_grounded and move_input == self.wall_direction:
            self.animator.current_animation = "Wall Jump"
            # Bônus: você pode aplicar um "drag" no Rigidbody aqui para ele escorregar devagar

        elif not self.is_grounded:
            if self.rb.velocity.y < 0:
                self.animator.current_animation = "Jump"
            else:
                self.animator.current_animation = "Fall"

        else:  # Está no chão
            if abs(self.rb.velocity.x) > 0.1:
                self.animator.current_animation = "Run"
            else:
                self.animator.current_animation = "Idle"