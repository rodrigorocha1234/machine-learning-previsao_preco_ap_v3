import time
import urllib.error
import urllib.request
import json

from app_build.observabilidade_metricas.emissor_logs import EmissorLogs, EventoLog


def main() -> None:
    emissor = EmissorLogs()

    eventos = [
        # DEBUG / TRACE
        EventoLog(
            nivel="TRACE",
            servico="mlflow-serving",
            mensagem="[TRACE] Entrada do payload de inferência recebida com 2 registros (452 bytes).",
        ),
        EventoLog(
            nivel="DEBUG",
            servico="mlflow-serving",
            mensagem="[DEBUG] Validando schema do DataFrame: colunas ['Area_Privativa_m2', 'Quartos', 'Suites', 'Vagas', 'Banheiros', 'Zona', 'Bairro'].",
        ),
        EventoLog(
            nivel="DEBUG",
            servico="mlflow-server",
            mensagem="[DEBUG] Consulta de metadados do experimento 2 executada em 2.4ms no PostgreSQL.",
        ),
        EventoLog(
            nivel="DEBUG",
            servico="app_pipeline",
            mensagem="[DEBUG] Pipeline Scikit-Learn executando imputação numérica por mediana para fold 14.",
        ),
        
        # INFO / NOTICE
        EventoLog(
            nivel="NOTICE",
            servico="alloy",
            mensagem="[NOTICE] Componente loki.source.docker recarregou regras de relabel com 8 alvos ativos.",
        ),
        EventoLog(
            nivel="INFO",
            servico="mlflow-serving",
            mensagem="[INFO] Requisição POST /invocations atendida com sucesso. Predições calculadas em 14.8ms.",
        ),
        EventoLog(
            nivel="INFO",
            servico="mlflow-server",
            mensagem="[INFO] Modelo 'previsao_preco_apartamento_modelo' versão 1 vinculado ao alias @champion.",
        ),
        EventoLog(
            nivel="INFO",
            servico="app_pipeline",
            mensagem="[INFO] Testes estatísticos de Friedman (p < 0.001) e Nemenyi concluídos. Random Forest selecionado campeão.",
        ),
        EventoLog(
            nivel="INFO",
            servico="prometheus",
            mensagem="[INFO] Scrape da telemetria de MLflow Serving realizado com sucesso (status=200).",
        ),
        
        # WARNING / WARN
        EventoLog(
            nivel="WARNING",
            servico="app_pipeline",
            mensagem="[WARN] Bairro 'Recreio das Acácias' com amostragem insuficiente (N=9 < 15). Ativado fallback regional para 'Zona Sul'.",
        ),
        EventoLog(
            nivel="WARNING",
            servico="mlflow-serving",
            mensagem="[WARN] Latência de inferência percentil 95 atingiu 78ms, aproximando-se do limiar de alerta SLA (100ms).",
        ),
        EventoLog(
            nivel="WARNING",
            servico="mlflow-postgres",
            mensagem="[WARN] Pool de conexões atingiu 70% de capacidade durante busca exaustiva de hiperparâmetros.",
        ),
        
        # ERROR / ERR
        EventoLog(
            nivel="ERROR",
            servico="mlflow-serving",
            mensagem="[ERROR] Falha de validação de contrato: Payload HTTP recebido sem o campo obrigatório 'Area_Privativa_m2'. Retornado HTTP 400 Bad Request.",
        ),
        EventoLog(
            nivel="ERROR",
            servico="app_pipeline",
            mensagem="[ERROR] Tentativa de carregamento de arquivo de holdout com chave de acesso incorreta. Acesso rejeitado pelo cofre lógico.",
        ),
        EventoLog(
            nivel="ERROR",
            servico="storage",
            mensagem="[ERROR] Erro de timeout transitório ao transferir artefato de modelo temporário (recuperado automaticamente com retry).",
        ),
        
        # CRITICAL / FATAL
        EventoLog(
            nivel="CRITICAL",
            servico="app_pipeline",
            mensagem="[CRITICAL] Violação de Integridade Detectada: Divergência no checksum SHA-256 da base de holdout selada! Operação abortada preventivamente.",
        ),
        EventoLog(
            nivel="CRITICAL",
            servico="mlflow-serving",
            mensagem="[CRITICAL] Falha crítica de inicialização do backend PyFunc: Modelo ausente no artefato storage S3. Reinicialização de segurança acionada.",
        ),
        EventoLog(
            nivel="FATAL",
            servico="mlflow-server",
            mensagem="[FATAL] Perda súbita de heartbeat com o nó de armazenamento primário. Transição emergencial para modo somente-leitura ativada.",
        ),
    ]

    total = emissor.emitir_lote(eventos)
    print(f"Sucesso: {total} eventos de logs de telemetria emitidos para o Grafana Loki cobrindo todos os níveis!")


if __name__ == "__main__":
    main()
