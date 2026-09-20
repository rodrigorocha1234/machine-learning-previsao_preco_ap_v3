"""Ponto de entrada para o treinamento do pipeline de Machine Learning."""

from pathlib import Path
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from ..configuracao_pkg.leitor_configuracao import LeitorConfiguracao
from ..pipeline_pkg.pipeline_treinamento import PipelineTreinamento
from ..pipeline_pkg.resultado_treinamento import ResultadoTreinamento
from ..modelos_pkg.regressor_protocol import RegressorProtocol



class Treinar:
    """Orquestrador do ponto de entrada de treinamento."""

    def __init__(self, caminho_configuracao: str | Path = "configuracao.yaml") -> None:
        self._caminho_config = Path(caminho_configuracao)
        self._leitor = LeitorConfiguracao()

    def executar_treinamento(
        self,
        caminho_dados: str | Path = "dados/bruto/imoveis.csv",
    ) -> ResultadoTreinamento[RegressorProtocol]:
        """Dispara o pipeline completo de treinamento conforme a configuração.

        Parameters
        ----------
        caminho_dados : str | Path
            Caminho para a base de dados bruta.

        Returns
        -------
        ResultadoTreinamento[RegressorProtocol]
            Resultado do modelo campeão treinado e registrado.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        config = self._leitor.carregar_do_arquivo(self._caminho_config)
        pipeline = PipelineTreinamento(configuracao=config)
        return pipeline.executar(caminho_dados)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Executa o pipeline completo de treinamento de previsão de preço de imóveis."
    )
    parser.add_argument(
        "--dados",
        type=str,
        default="dados/bruto/imoveis.csv",
        help="Caminho para o arquivo de dados de entrada (CSV ou Excel). Padrão: dados/bruto/imoveis.csv",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configuracao.yaml",
        help="Caminho para o arquivo YAML de configuração. Padrão: configuracao.yaml",
    )
    args = parser.parse_args()

    app = Treinar(caminho_configuracao=args.config)
    app.executar_treinamento(caminho_dados=args.dados)

