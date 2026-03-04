import raylibpy as rl

from EasyCells3D import Game, Vec2
from EasyCells3D.Components import Item, Animation2D, Sprite, Animator2D, Component
from EasyCells3D.PhysicsComponents import Rigidbody, Collider, RectCollider
from UserComponents.platform.Input import Input, Action, ControllerType

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

        # Cria o objeto de animação e adiciona ao dicionário
        dict_animations[anim_name] = Animation2D(
            speed=0.05,
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
    player_item.AddComponent(RectCollider(rl.Rectangle(0, 0, 22, 28), debug=True, mask=0))
    player_item.AddComponent(Rigidbody(use_gravity=True, restitution=0))
    player_item.AddComponent(Sprite(final_texture, size=(frame_size, frame_size)))
    player_item.AddComponent(Animator2D(dict_animations, "Idle"))
    player_item.AddComponent(Player())

    return player_item


class Player(Component):
    def __init__(self):
        self.rb: Rigidbody = None
        self.animator: Animator2D = None
        self.sprite: Sprite = None
        self.collider: Collider = None

        # Variáveis de movimento
        self.move_speed = 100.0
        self.jump_force = 400.0
        self.wall_jump_force = Vec2(550.0, 400.0)  # Força (X, Y) ao pular da parede
        self.wall_slide_speed = 100.0  # Velocidade limite (drag) ao escorregar

        self.ray_length_down = 20.0
        self.ray_length_side = 15.0

        # Máscara do cenário com a qual os raycasts vão colidir
        self.environment_mask = 1

        # Variáveis de estado
        self.can_double_jump = False
        self.is_hit = False
        self.is_grounded = False
        self.is_touching_wall = False
        self.wall_direction = 0

        self.input = Input(ControllerType.KEYBOARD)

    def init(self):
        self.rb = self.GetComponent(Rigidbody)
        self.animator = self.GetComponent(Animator2D)
        self.sprite = self.GetComponent(Sprite)
        self.collider = self.GetComponent(Collider)

    def loop(self):
        if self.is_hit:
            # Trava o personagem na animação de hit
            if self.animator.current_animation != "Hit":
                self.is_hit = False
            else:
                return

        self._check_environment()
        self._handle_movement()
        self._update_animations()

    def take_hit(self):
        self.is_hit = True
        self.animator.current_animation = "Hit"

    def _check_environment(self):
        # Origem do raio é o centro do jogador no mundo global
        pos = self.transform.positionVec2

        # --- Raycast para o Chão ---
        # Ajuste da distância para o RectCast (considerando que o Rect já tem o tamanho do player)
        cast_distance = max(0.1, self.ray_length_down - (frame_size / 2))
        hit_ground = Collider.rect_cast_static(
            origin=pos,
            size=Vec2(20, frame_size),
            angle=0,
            direction=Vec2(0, 1),  # Para baixo
            max_distance=cast_distance,
            mask=self.environment_mask
        )

        self.is_grounded = False
        # hit_ground retorna (Collider, Vec2 (Ponto), Vec2 (Normal), float (Dist)) se atingir
        if hit_ground:
            col, point, normal, _ = hit_ground
            # Ignora colisão com o próprio Collider do Player
            if col != self.collider:
                self.is_grounded = True
                self.can_double_jump = True

        # --- Raycast para as Paredes ---
        self.is_touching_wall = False
        self.wall_direction = 0

        # Direita
        hit_right = Collider.ray_cast_static(
            origin=pos, direction=Vec2(1, 0), max_distance=self.ray_length_side, mask=self.environment_mask
        )
        if hit_right and hit_right[0] != self.collider:
            self.is_touching_wall = True
            self.wall_direction = 1

        # Esquerda
        hit_left = Collider.ray_cast_static(
            origin=pos, direction=Vec2(-1, 0), max_distance=self.ray_length_side, mask=self.environment_mask
        )
        if hit_left and hit_left[0] != self.collider:
            self.is_touching_wall = True
            self.wall_direction = -1

    def _handle_movement(self):
        move_input = self.input.get_axis().x

        if move_input > 0:
            self.sprite.horizontal_flip = False
        elif move_input < 0:
            self.sprite.horizontal_flip = True

        # Aplica a velocidade horizontal (Move-se normalmente)
        self.rb.velocity.x = move_input * self.move_speed

        # --- Lógica de Wall Slide (Drag) ---
        is_wall_sliding = False
        # Para agarrar na parede: toca a parede, está no ar, segurando para o lado da parede, e caindo (velocidade.y > 0)
        if self.is_touching_wall and not self.is_grounded and (move_input * self.wall_direction > 0.1) and self.rb.velocity.y > 0:
            is_wall_sliding = True
            # Aplica o "Drag" fixando um limite máximo de queda
            if self.rb.velocity.y > self.wall_slide_speed:
                self.rb.velocity.y = self.wall_slide_speed

        # --- Lógica de Pulo ---
        if self.input.is_action_just_pressed(Action.JUMP):
            if self.is_grounded:
                # Pulo Normal
                self.rb.velocity.y = -self.jump_force
                self.animator.current_animation = "Jump"

            elif is_wall_sliding or (self.is_touching_wall and not self.is_grounded):
                # Wall Jump: Aplica força vertical e afasta o player da parede horizontalmente
                self.rb.velocity.y = -self.wall_jump_force.y
                self.rb.velocity.x = -self.wall_direction * self.wall_jump_force.x
                self.animator.current_animation = "Wall Jump"

                # Vira a sprite automaticamente para o lado que está pulando
                self.sprite.horizontal_flip = (self.wall_direction > 0)

            elif self.can_double_jump:
                # Double Jump
                self.rb.velocity.y = -self.jump_force
                self.can_double_jump = False
                self.animator.current_animation = "Double Jump"

    def _update_animations(self):
        # Não sobrescrever animações que já foram engatilhadas e devem durar um pouco
        if self.animator.current_animation == "Hit": return
        if self.animator.current_animation == "Double Jump" and self.rb.velocity.y < 0: return
        if self.animator.current_animation == "Wall Jump" and self.rb.velocity.y < 0: return

        move_input = self.input.get_axis().x

        # Máquina de estados
        if self.is_touching_wall and not self.is_grounded and (move_input * self.wall_direction > 0.1) and self.rb.velocity.y > 0:
            self.animator.current_animation = "Wall Jump"
        elif not self.is_grounded:
            if self.rb.velocity.y < 0:
                self.animator.current_animation = "Jump"
            else:
                self.animator.current_animation = "Fall"
        else:  # No chão
            if abs(move_input) > 0.1:
                self.animator.current_animation = "Run"
            else:
                self.animator.current_animation = "Idle"