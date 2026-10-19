# Rede: uso e contratos

O módulo mantém TCP/UDP, NetworkManager, NetworkComponent, NetworkVariable e NetworkTransform.
NetworkObject identifica a raiz criada por spawn. Use a mesma versão da biblioteca e as mesmas
factories/classes nos peers. O handshake TCP/UDP mudou; versões antigas não são compatíveis.

## Ambiente e testes

No PowerShell, a partir da raiz do repositório:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

A .venv é local e ignorada pelo Git. A suíte inclui testes com três processos e sockets em loopback;
não abre janela de jogo. PyBullet pode exigir ferramentas de compilação se não houver wheel no ambiente.

## Spawn e IDs automáticos

Defina as classes e a factory em um módulo compartilhado por servidor e clientes:

~~~python
from EasyCells3D.NetworkComponents import (
    NetworkComponent, NetworkManager, NetworkObject,
    NetworkTransform, NetworkVariable, Rpc, SendTo,
)

class Player(NetworkComponent):
    def __init__(self):
        super().__init__()  # identifier e owner serão atribuídos pelo servidor
        self.health = NetworkVariable(100)

    def on_network_spawn(self):
        # Component.init já terminou e o estado inicial foi aplicado.
        print(self.identifier, self.owner, self.health.value)

    @Rpc(send_to=SendTo.ALL)
    def show_message(self, *, text):
        print(text)

def player_factory(game, *, x=0):
    item = game.CreateItem()
    item.transform.x = x
    item.AddComponent(Player())
    item.AddComponent(NetworkTransform())
    return item

def start_network(game, is_server, ip="127.0.0.1"):
    manager = NetworkManager(ip, 25765, is_server)
    manager.register_prefab("player", player_factory)
    if is_server:
        manager.connect_callbacks.append(
            lambda client_id: manager.spawn("player", owner=client_id, x=2)
        )
    game.CreateItem().AddComponent(manager)
    return manager
~~~

Chame start_network no init do level. O construtor só configura o manager; sockets e threads começam
em Component.init. Há um NetworkManager inicializado por processo. Registre as factories antes da conexão.
spawn exige um manager inicializado e só pode ser chamado no servidor.

- A factory recebe game e retorna um novo Item raiz; deve montar a hierarquia e os componentes antes de retornar.
- Filhos irmãos precisam de nomes distintos e estáveis, ou IDs de cena distintos. A estrutura e as classes devem coincidir nos peers.
- O servidor atribui IDs aos objetos e componentes. NetworkVariable declarada como atributo direto do componente,
  no construtor, recebe identidade pelo componente e nome do atributo e herda seu owner, salvo owner explícito.
- IDs explícitos continuam disponíveis para objetos de cena e variáveis independentes. Duplicatas geram ValueError.
- O init normal continua no ciclo da engine. on_network_spawn é o ponto para código que precisa do estado inicial.
- A entrada tardia recebe objetos existentes, transformações e NetworkVariables atuais. Atributos Python comuns
  e o histórico de RPCs não são replicados automaticamente.
- Argumentos de factory/RPC e valores replicados precisam ser dados aceitos pelo codec: números, strings,
  bytes, listas, tuplas, dicionários e outros tipos básicos. Vec3/Quaternion e objetos personalizados não são
  enviados diretamente; converta-os em tuplas. NetworkTransform já faz sua própria serialização.

No servidor:

~~~python
item = manager.spawn("player", owner=client_id, x=10)
network_id = item.GetComponent(NetworkObject).identifier
manager.despawn(item)        # ou manager.despawn(network_id)
# item.Destroy() no servidor também replica a destruição.
~~~

manager.spawned retorna um mapa de IDs para Items. Clientes recebem PermissionError ao chamar spawn/despawn.
As factories são uma lista local permitida: nomes enviados pela rede não provocam imports dinâmicos.
A política para objetos de um jogador desconectado pertence ao jogo; use disconnect_callback para despawn,
se desejado. Para persistência entre cenas, marque a raiz com destroy_on_load=False na factory.

## RPCs

| Destino | Quem executa |
|---|---|
| ALL | Servidor e todos os clientes, incluindo o originador |
| SERVER | Somente o servidor |
| CLIENTS | Todos os clientes, sem executar no servidor |
| OWNER | Somente o dono, mesmo quando ele originou a chamada |
| NOT_ME | Todos, exceto o originador |

Clientes enviam ao servidor para autorização e retransmissão. Sua execução local, quando aplicável,
acontece ao receber a resposta; não há execução preditiva automática. O servidor executa localmente
quando pertence ao destino. require_owner=True valida o remetente no servidor; o servidor mantém autoridade.
RPCs síncronos (`def`) mantêm o comportamento de envio sem resposta remota. RPCs `async def`
retornam um `RpcCall` aguardável pelo Scheduler, conforme descrito abaixo.

Argumentos posicionais e nomeados são preservados, inclusive keyword-only. RPCs aninhados obedecem
seus próprios destinos. call_rpc_on_client(client_id, method, *args, **kwargs) é um envio explícito
feito pelo servidor a um cliente e prevalece sobre o destino normal do decorador.

Funções livres e staticmethod usam module.qualname, evitando colisões entre classes/módulos. Ambas as
ordens de Rpc e staticmethod são aceitas. Métodos de instância não entram no registro global. Para uma
função global aceitar chamadas de clientes, use require_owner=False e valide suas regras no servidor.

### RPCs assíncronos e Scheduler

~~~python
from EasyCells3D.NetworkComponents import NetworkComponent, Rpc, SendTo, RpcRemoteError

class Player(NetworkComponent):
    @Rpc(send_to=SendTo.OWNER, require_owner=False, timeout=5.0)
    async def consultar_pontos(self, *, multiplicador=1):
        await self.game.scheduler.next_frame()
        return self.pontos * multiplicador

async def consultar(game, player):
    try:
        pontos = await player.consultar_pontos(multiplicador=2)
        print(pontos)
    except RpcRemoteError as erro:
        print(erro.peer_id, erro.remote_type, erro.message)

# No código de gameplay, com o manager conectado e o componente inicializado:
game.scheduler.create_task(consultar(game, player))
~~~

`player.consultar_pontos(...)` envia imediatamente e devolve um `RpcCall`: você pode guardá-lo,
aguardá-lo depois ou enviar sem esperar. Diferentemente de uma coroutine Python comum, não é
preciso passá-lo a `create_task` para iniciar o envio. Use `create_task` para a coroutine que faz
o `await`. `RpcCall` pertence ao Scheduler daquele jogo; não é um `asyncio.Future`.

O percurso entre clientes é cliente A → servidor → cliente B → servidor → cliente A. O servidor
valida a chamada e encaminha as respostas; apenas os destinatários executam o corpo. Os handlers
rodam no Scheduler da thread principal, incluindo a execução local no servidor, e podem aguardar
outros RPCs, tarefas, `sleep` ou `next_frame`. Uma tarefa suspensa não impede outros RPCs de começar.
O servidor encaminha a chamada antes de executar seu próprio handler, quando também for destinatário.

| Chamada | Resultado do `await` |
|---|---|
| `SERVER`, `OWNER` | Valor retornado pelo único destinatário, inclusive `None` |
| `ALL`, `CLIENTS`, `NOT_ME` | Dicionário `{peer_id: resultado}`, após todos terminarem |
| `manager.call_rpc_on_client(id, metodo, ...)` | Valor do cliente escolhido; somente o servidor pode usar essa API |

O servidor tem `peer_id=0`. Destinos coletivos sempre retornam dicionário, mesmo com um único
participante; sem destinatários retornam `{}`. Um destino único indisponível gera `RpcConnectionError`.
Os destinatários são fixados quando o servidor aceita a chamada. Entradas posteriores e mudanças
de owner não alteram essa chamada; desconexões dos destinatários pendentes geram falhas.
`require_owner=True` continua como padrão: para A chamar o `OWNER` de um objeto de B, o RPC deve
permitir chamadas de outros clientes com `require_owner=False`. O servidor mantém autoridade.

- RPCs assíncronos exigem **TCP**; `@Rpc(protocol=Protocol.UDP)` em uma função `async def` gera `ValueError`.
  Use a mesma versão do código nos peers. O protocolo assíncrono não é entendido por versões anteriores.
- Configure o prazo padrão com `NetworkManager(..., rpc_timeout=10.0)`, ou sobrescreva por função com
  `@Rpc(..., timeout=5.0)`. Valores precisam ser positivos e finitos. O prazo começa na chamada,
  inclui ida e volta e usa relógio monotônico, independente de `game.run_time`. A entrega de resultados
  e a detecção de timeout dependem da atualização do loop; não ocorrem enquanto a engine estiver parada.
- Retornos precisam ser dados aceitos pelo codec e caber no limite de pacote TCP. Instâncias de
  componentes/classes personalizadas e retornos grandes demais viram erros, sem enviar objetos executáveis.
- Uma exceção remota vira `RpcRemoteError`, com `peer_id`, `remote_type` e `message`; a classe original
  não é reconstruída. Falta de permissão, componente ou método também produz erro explícito.
  `RpcTimeoutError` e `RpcConnectionError` são subclasses de `RpcRemoteError` e também de
  `TimeoutError` e `ConnectionError`, respectivamente.
- Um coletivo com falhas aguarda os demais destinatários ou o prazo e lança `RpcGroupError`.
  `erro.results` preserva os retornos recebidos e `erro.errors` contém `{peer_id: erro}`. Antes de
  receber a confirmação dos destinatários, falhas de autorização/conexão/timeout são erros diretos.
- `chamada.done` indica conclusão; `chamada.result()` obtém o retorno ou lança a exceção. Falhas
  que não foram observadas por `await`/`result()` são registradas pelo logger
  `EasyCells3D.NetworkComponents.AsyncRpc` quando o handle é descartado.
- `chamada.cancel()` cancela apenas a espera local, com `SchedulerTaskCancelled` (de
  `EasyCells3D.scheduler`). Cancelar uma tarefa que aguarda a chamada remove aquela tarefa da espera;
  outros consumidores do mesmo handle podem continuar. Nenhuma dessas ações interrompe o handler remoto.
- Timeout não desfaz efeitos remotos e não repete a chamada. Respostas tardias ou duplicadas são ignoradas.
  A execução remota pode continuar após o prazo ou a desconexão de quem chamou.
- A troca de cena (`scheduler.clear()`) cancela chamadas locais pendentes e execuções locais, incluindo
  funções globais/estáticas. Destruir um NetworkComponent cancela seus handlers locais; sobrescritas de
  `on_destroy` devem chamar `super().on_destroy()`. O chamador recebe a falha se a conexão ainda existir.
  Encerrar o manager também encerra suas chamadas e execuções. O transporte persistente não transforma
  os RPCs em tarefas persistentes entre cenas.

~~~python
from EasyCells3D.NetworkComponents import RpcGroupError

async def consultar_todos(player):
    try:
        resultados = await player.consultar_todos()  # RPC declarado com SendTo.ALL
    except RpcGroupError as erro:
        print("Resultados recebidos:", erro.results)
        print("Falhas por participante:", erro.errors)
~~~

Internamente, `OP_ASYNC_RPC=5` usa `(op, call_id, tipo, payload)` para requisição, aceitação,
execução, resposta e rejeição. O servidor correlaciona cada resposta com o remetente e a chamada
pendente, sem confiar em um ID de participante declarado pelo cliente. Resultados coletivos são
encaminhados individualmente, permitindo preservar respostas parciais até o timeout.

## Variáveis, transform e callbacks

Escreva por variable.value = novo_valor. Quando require_owner=True, um cliente que não é dono recebe
PermissionError antes de alterar o valor ou enviar o pacote. O servidor pode escrever qualquer variável;
require_owner=False permite escrita compartilhada. Uma tentativa remota indevida é rejeitada, e o servidor
devolve o valor válido. Mutações internas de listas/dicionários não disparam replicação: atribua um novo valor.

Variáveis explícitas criadas antes da conexão aguardam o TCP para consultar o estado inicial. Para variáveis
automáticas, espere on_network_spawn antes de escrever.

NetworkTransform usa UDP. sync_frequency continua sendo um intervalo em segundos, não Hz. O envio roda
no loop do componente: enable=False suspende os envios e a troca de cena não cancela uma coroutine de rede.
A suavização existente é de posição; não há predição/reconciliação de física.

Callbacks de conexão dos transportes são entregues em read/poll_events, na thread que faz o polling.
NetworkManager faz isso no loop da engine. Usando classes de transporte diretamente, faça o polling na
thread principal. Callbacks não dependem de tarefas do scheduler que possam ser apagadas ao trocar cenas.

## Autenticação e limites

O TCP atribui um ID e uma chave aleatória de sessão. O UDP exige essa chave e autentica identidade, direção,
sequência e payload com HMAC-SHA256 antes de desserializar. Uma janela de 64 sequências aceita reordenação
limitada e rejeita repetição. IDs TCP não são reciclados na sessão, evitando que novos peers herdem ownership.

Essa autenticação não cifra o tráfego: o TCP não usa TLS. Ela não protege contra alguém capaz de interceptar
a entrega da chave TCP. Tampouco substitui validação de gameplay no servidor. O unpickler bloqueia resolução
de classes/funções; ainda é necessário tratar os valores recebidos como dados não confiáveis.

| Limite | Padrão / configuração |
|---|---|
| Clientes TCP simultâneos | max_clients=64 |
| IDs de conexão por sessão do servidor | 65.535; depois, reinicie a sessão |
| Payload TCP | 1 MiB |
| Datagrama UDP completo | 1.200 bytes, incluindo cabeçalho e MAC |
| Fila UDP por cliente | max_udp_queue=128; descarta os mais antigos quando cheia |
| Recepção UDP autenticada | 1.000 pacotes por segundo por sessão/direção |
| Pacotes processados por frame | max_packets_per_frame=256 |
| Pacotes por peer por frame, somando protocolos | max_packets_per_peer=32 |
| Orçamento de processamento | max_frame_ms=2.0 |
| Objetos criados por spawn simultaneamente | max_spawned_objects=1024 |

Os parâmetros max_* acima pertencem a NetworkManager, exceto os limites fixos explicitados.
O orçamento de tempo é verificado entre pacotes; não interrompe um RPC já em execução nem um sendall TCP.
Mensagens maiores que o limite UDP devem usar TCP. NetworkTransform requer enable_udp=True.

Para usar UDP diretamente, forneça os provedores de ID/chave da sessão TCP. UdpTransport(..., tcp=...)
faz essa associação; UDP sem credenciais não é aceito. Encerre os transportes com close().
