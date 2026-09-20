"""Testes unitários para leitura e tipagem da configuração."""

from pathlib import Path
from imobiliaria_ml.configuracao_pkg.leitor_configuracao import LeitorConfiguracao
from imobiliaria_ml.enums_pkg.tipo_metrica import TipoMetrica
from imobiliaria_ml.enums_pkg.tipo_ensemble import TipoEnsemble


def test_leitor_configuracao_carrega_arquivo_padrao() -> None:
    leitor = LeitorConfiguracao()
    config = leitor.carregar_do_arquivo("configuracao.yaml")

    assert config.alvo == "Valor_da_Venda"
    assert config.seed == 42
    assert config.total_passos == 12
    assert config.test_size == 0.20
    assert config.n_splits == 5
    assert config.n_repeats == 30
    assert config.metrica_primaria == TipoMetrica.RMSE
    assert TipoEnsemble.VOTING_MEDIA in config.tecnicas_ensemble


def test_leitor_configuracao_carrega_test_size_customizado(tmp_path: Path) -> None:
    yaml_conteudo = """
projeto:
  test_size: 0.30
  seed: 123
"""
    arquivo_teste = tmp_path / "custom.yaml"
    arquivo_teste.write_text(yaml_conteudo, encoding="utf-8")

    leitor = LeitorConfiguracao()
    config = leitor.carregar_do_arquivo(arquivo_teste)

    assert config.test_size == 0.30
    assert config.seed == 123

