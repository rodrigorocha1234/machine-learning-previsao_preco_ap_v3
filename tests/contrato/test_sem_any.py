"""Teste de contrato estrito para garantir ausência total de Any em src/."""

import ast
from pathlib import Path
import re


def test_proibicao_estrita_de_any_em_src() -> None:
    """Varre todos os arquivos Python em src/ e assegura que Any não é utilizado."""
    raiz_src = Path(__file__).resolve().parent.parent.parent / "src"
    arquivos_py = list(raiz_src.rglob("*.py"))

    assert len(arquivos_py) > 0, "Nenhum arquivo Python encontrado em src/"

    padroes_proibidos = [
        re.compile(r"from\s+typing\s+import\s+.*\bAny\b"),
        re.compile(r"\btyping\.Any\b"),
        re.compile(r"\bdict\[\s*str\s*,\s*Any\s*\]"),
        re.compile(r"\blist\[\s*Any\s*\]"),
        re.compile(r"\btuple\[\s*Any\s*,\s*\.\.\.\s*\]"),
    ]

    violacoes: list[str] = []

    for caminho in arquivos_py:
        with open(caminho, "r", encoding="utf-8") as f:
            conteudo = f.read()

        # Checagem via AST para imports e anotações
        try:
            arvore = ast.parse(conteudo, filename=str(caminho))
            for no in ast.walk(arvore):
                # Detecta import Any
                if isinstance(no, ast.ImportFrom) and no.module == "typing":
                    for alias in no.names:
                        if alias.name == "Any":
                            violacoes.append(f"{caminho.name}: import Any da biblioteca typing")
                elif isinstance(no, ast.Import):
                    for alias in no.names:
                        if alias.name == "typing.Any":
                            violacoes.append(f"{caminho.name}: import typing.Any direto")
                # Detecta Name(id='Any') em anotações
                elif isinstance(no, ast.Name) and no.id == "Any":
                    violacoes.append(f"{caminho.name}: uso de identificador Any (linha {no.lineno})")
        except SyntaxError as e:
            violacoes.append(f"{caminho.name}: Erro de sintaxe ao analisar AST: {e}")

        # Checagem via Regex em linhas de código (ignorando comentários)
        linhas = conteudo.splitlines()
        for num_linha, linha in enumerate(linhas, start=1):
            linha_sem_comentario = linha.split("#")[0].strip()
            for padrao in padroes_proibidos:
                if padrao.search(linha_sem_comentario):
                    violacoes.append(
                        f"{caminho.name}:{num_linha} - Padrão proibido detectado: '{linha.strip()}'"
                    )

    assert not violacoes, "Violações de proibição de Any encontradas:\n" + "\n".join(violacoes)
