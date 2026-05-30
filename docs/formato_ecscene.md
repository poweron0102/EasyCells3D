# Formato `.ecscene`

O `.ecscene` e um arquivo JSON carregado por `SceneLoader`. O formato suportado atualmente e apenas a versao 2:

```json
{
  "format": "easycells3d.scene",
  "version": 2,
  "assets": {},
  "Item": []
}
```

Arquivos precisam usar a extensao `.ecscene`. O campo `Item` e obrigatorio e deve ser uma lista.

## Estrutura da Cena

Campos da raiz:

| Campo | Obrigatorio | Tipo | Descricao |
| --- | --- | --- | --- |
| `format` | Sim | string | Deve ser exatamente `"easycells3d.scene"`. |
| `version` | Sim | number | Deve ser `2`. |
| `assets` | Nao | object | Assets globais reutilizaveis por componentes. |
| `Item` | Sim | array | Lista de objetos da cena. |

Todos os ids de `Item` e de componente devem ser strings decimais UInt128. Eles precisam ser unicos na cena inteira.

## Items

Cada item representa um objeto da cena:

```json
{
  "id": "1",
  "name": "Root",
  "parent": null,
  "enabled": true,
  "transform": {
    "position": [0, 0, 0],
    "rotation": [0, 0, 0],
    "scale": [1, 1, 1]
  },
  "components": []
}
```

Campos do item:

| Campo | Obrigatorio | Tipo | Descricao |
| --- | --- | --- | --- |
| `id` | Sim | string | Id decimal UInt128 do item. |
| `name` | Sim | string | Nome runtime do item. Nao e usado para referencias. |
| `parent` | Sim | string ou null | Id do item pai, ou `null` para raiz. |
| `enabled` | Nao | bool | Padrao: `true`. |
| `transform` | Nao | object | Transform local do item. |
| `components` | Nao | array | Componentes anexados ao item. |

`transform.position`, `transform.rotation` e `transform.scale` devem ser arrays com 3 numeros. Rotacao e escrita em graus e convertida para quaternion no carregamento.

Valores padrao do transform:

```json
{
  "position": [0, 0, 0],
  "rotation": [0, 0, 0],
  "scale": [1, 1, 1]
}
```

## Componentes

Componentes sao criados pelo `ComponentRegistry` usando o campo `type` e os argumentos de construtor em `args`:

```json
{
  "id": "2",
  "type": "PlayerController",
  "enabled": true,
  "args": {
    "speed": 8.0
  },
  "fields": {
    "target": {"$ref": "3"}
  }
}
```

Campos do componente:

| Campo | Obrigatorio | Tipo | Descricao |
| --- | --- | --- | --- |
| `id` | Sim | string | Id decimal UInt128 do componente. |
| `type` | Sim | string | Nome registrado do componente ou caminho completo registrado. |
| `enabled` | Nao | bool | Padrao: `true`. |
| `args` | Nao | object | Argumentos passados para o construtor. |
| `fields` | Nao | object | Valores aplicados depois que todos os componentes foram criados. |

Referencias para items ou componentes da cena nao sao permitidas em `args`, porque os objetos ainda estao sendo criados. Use `fields` para referencias de cena.

Quando um campo declarado com `SerializeField` existe no componente, o loader converte valores simples para o tipo declarado nos casos `bool`, `int`, `float`, `str` e `string`.

## Assets

Assets globais ficam no campo `assets` da raiz:

```json
{
  "assets": {
    "terrain": {
      "type": "TileMapAsset",
      "args": {
        "img_path": "Tilesets/terrain.png",
        "tile_size_w": 16,
        "tile_size_h": 16
      }
    }
  }
}
```

Um componente pode usar um asset global com `$assetRef`:

```json
{
  "args": {
    "asset": {"$assetRef": "terrain"}
  }
}
```

Tambem e possivel criar asset inline usando `type` e `args`:

```json
{
  "args": {
    "inline": {
      "type": "TileMapAsset",
      "args": {
        "img_path": "Tilesets/inline.png",
        "tile_size_w": 8,
        "tile_size_h": 8
      }
    }
  }
}
```

Se o asset tiver metodos exportados com `@export`, um selector pode chamar um metodo permitido e usar o retorno como valor:

```json
{
  "$assetRef": "terrain",
  "selector": {
    "method": "get_tile",
    "args": {"x": 3, "y": 4}
  }
}
```

`selector.args` pode ser objeto, para chamada com argumentos nomeados, ou lista, para chamada posicional. O metodo precisa ser exportado; metodos comuns nao podem ser chamados pelo arquivo de cena.

## Referencias

Referencia para item:

```json
{
  "fields": {
    "target": {"$ref": "3"}
  }
}
```

Referencia para componente:

```json
{
  "fields": {
    "target_component": {"$componentRef": "4"}
  }
}
```

Regras:

- `$ref` aponta sempre para `Item.id`.
- `$componentRef` aponta sempre para `Component.id`.
- Referencias por `name` nao sao suportadas.

## Exemplo Completo

```json
{
  "format": "easycells3d.scene",
  "version": 2,
  "assets": {
    "terrain": {
      "type": "TileMapAsset",
      "args": {
        "img_path": "Tilesets/terrain.png",
        "tile_size_w": 16,
        "tile_size_h": 16
      }
    }
  },
  "Item": [
    {
      "id": "1",
      "name": "Root",
      "parent": null,
      "transform": {
        "position": [0, 0, 0],
        "rotation": [0, 0, 0],
        "scale": [1, 1, 1]
      },
      "components": [
        {
          "id": "2",
          "type": "Recorder",
          "args": {
            "asset": {"$assetRef": "terrain"},
            "tile": {
              "$assetRef": "terrain",
              "selector": {
                "method": "get_tile",
                "args": {"x": 3, "y": 4}
              }
            }
          },
          "fields": {
            "target": {"$ref": "3"},
            "target_component": {"$componentRef": "4"}
          }
        }
      ]
    },
    {
      "id": "3",
      "name": "Child",
      "parent": "1",
      "components": [
        {
          "id": "4",
          "type": "Recorder",
          "args": {},
          "fields": {}
        }
      ]
    }
  ]
}
```
