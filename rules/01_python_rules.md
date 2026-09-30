# Rules 01 — Python

- Python 3.12+.
- Código próprio em português.
- Uma classe principal por arquivo `.py`; exceções: dataclasses, enums intimamente relacionados e exceções customizadas coesas.
- Módulos e pacotes próprios: exatamente duas palavras separadas por `_`.
- Não sombrear módulos externos como logging, json, typing, sklearn, pandas, numpy, mlflow, random, requests.
- Proibido `Any`.
- Usar Protocol, TypeVar, Generic, Self, Final, ClassVar, TypeAlias, Literal, `@override`, `@final`, `@overload`, `@runtime_checkable` quando semanticamente adequados.
- Usar `@property` e `@setter` para encapsular invariantes; não criar getter/setter cerimonial.
- Preferir composição.
- Exceções customizadas com nomes em português, atributos tipados, mensagem clara e exception chaining.
- Não capturar `Exception` indiscriminadamente.
- Código conciso: evitar classes, métodos e comentários redundantes.
