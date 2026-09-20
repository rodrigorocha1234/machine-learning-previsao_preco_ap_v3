"""Leitor de arquivos YAML com conversão automática para Enums de domínio."""

from pathlib import Path
import yaml
from ..enums_pkg.tipo_metrica import TipoMetrica
from ..enums_pkg.tipo_ensemble import TipoEnsemble
from ..enums_pkg.tipo_estimador_final import TipoEstimadorFinal
from .configuracao_projeto import ConfiguracaoProjeto


class LeitorConfiguracao:
    """Carrega o YAML de configuração e converte strings para instâncias de Enums."""

    def carregar_do_arquivo(self, caminho_yaml: str | Path) -> ConfiguracaoProjeto:
        """Lê o arquivo YAML e instancia ConfiguracaoProjeto tipada.

        Parameters
        ----------
        caminho_yaml : str | Path
            Caminho para o arquivo YAML.

        Returns
        -------
        ConfiguracaoProjeto
            Configuração validada e convertida em enums.
        """
        caminho = Path(caminho_yaml)
        if not caminho.exists():
            # Retorna configuração padrão se o arquivo não existir
            return ConfiguracaoProjeto()

        with open(caminho, "r", encoding="utf-8") as f:
            conteudo = yaml.safe_load(f) or {}

        proj = conteudo.get("projeto", {})
        val = conteudo.get("validacao", {})
        ens = conteudo.get("ensemble", {})
        mlf = conteudo.get("mlflow", {})
        neg = conteudo.get("negocio", {})
        drf = conteudo.get("drift", {})

        # Conversão de Enums
        metrica_str = str(val.get("metrica_primaria", "rmse")).lower()
        metrica = TipoMetrica(metrica_str) if metrica_str in TipoMetrica._value2member_map_ else TipoMetrica.RMSE

        tecnicas_str = ens.get("tecnicas_habilitadas", [])
        tecnicas: list[TipoEnsemble] = []
        for t in tecnicas_str:
            if str(t) in TipoEnsemble._value2member_map_:
                tecnicas.append(TipoEnsemble(str(t)))

        if not tecnicas:
            tecnicas = [
                TipoEnsemble.VOTING_MEDIA,
                TipoEnsemble.VOTING_PONDERADO_RMSE,
                TipoEnsemble.VOTING_PONDERADO_RANKING,
                TipoEnsemble.STACKING,
                TipoEnsemble.BAGGING,
            ]

        stacking_dict = ens.get("stacking", {})
        est_final_str = str(stacking_dict.get("estimador_final", "ridge")).lower()
        est_final = (
            TipoEstimadorFinal(est_final_str)
            if est_final_str in TipoEstimadorFinal._value2member_map_
            else TipoEstimadorFinal.RIDGE
        )

        bagging_dict = ens.get("bagging", {})

        return ConfiguracaoProjeto(
            alvo=str(proj.get("alvo", "Valor_da_Venda")),
            seed=int(proj.get("seed", 42)),
            n_splits=int(val.get("n_splits", 5)),
            n_repeats=int(val.get("n_repeats", 30)),
            alpha_friedman=float(val.get("alpha_friedman", 0.05)),
            alpha_nemenyi=float(val.get("alpha_nemenyi", 0.05)),
            metrica_primaria=metrica,
            usar_votacao=bool(ens.get("usar_votacao", False)),
            tecnicas_ensemble=tuple(tecnicas),
            stacking_estimador_final=est_final,
            bagging_n_estimators=int(bagging_dict.get("n_estimators", 100)),
            bagging_max_samples=float(bagging_dict.get("max_samples", 0.8)),
            bagging_bootstrap=bool(bagging_dict.get("bootstrap", True)),
            mlflow_tracking_uri_env=str(mlf.get("tracking_uri_env", "MLFLOW_TRACKING_URI")),
            mlflow_experimento=str(mlf.get("experimento", "previsao-preco-imoveis")),
            mlflow_nome_modelo=str(mlf.get("nome_modelo", "preco-imoveis")),
            mlflow_persistencia_local_artefatos=bool(mlf.get("persistencia_local_artefatos", False)),
            desconto_minimo=float(neg.get("desconto_minimo", 0.0)),
            desconto_maximo_automatico=float(neg.get("desconto_maximo_automatico", 5.0)),
            desconto_maximo_com_aprovacao=float(neg.get("desconto_maximo_com_aprovacao", 10.0)),
            drift_psi_atencao=float(drf.get("psi_atencao", 0.10)),
            drift_psi_forte=float(drf.get("psi_forte", 0.25)),
        )
