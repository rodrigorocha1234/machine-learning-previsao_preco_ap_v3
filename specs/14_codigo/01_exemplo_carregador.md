# Exemplo — Carregadores

```python
# carregador_base.py
from abc import ABC, abstractmethod

class CarregadorBase(ABC):
    @abstractmethod
    def carregar_dados(self, origem: str):
        ...
```

```python
# carregador_csv.py
import pandas as pd
from .carregador_base import CarregadorBase

class CarregadorCsv(CarregadorBase):
    def carregar_dados(self, origem: str):
        return pd.read_csv(origem)
```

```python
# fabrica_carregadores.py
class FabricaCarregadores:
    def criar(self, tipo: str):
        ...
```

Factory + Strategy permitem trocar CSV, Excel, banco ou storage sem alterar a pipeline.


## Enum do carregador

A `FabricaCarregadores` deve receber `TipoCarregador`:

```python
def criar(
    self,
    tipo: TipoCarregador
) -> CarregadorBase:
    ...
```

Não usar strings livres como `"csv"` ou `"excel"` internamente.
