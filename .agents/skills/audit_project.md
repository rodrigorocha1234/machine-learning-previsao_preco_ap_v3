# Skill — Audit Project

## Objetivo
Auditar e corrigir `app_build/` contra specs e rules.

## Gates bloqueantes
1. procurar `if`, `elif`, `match` em código próprio e exigir substituição por padrão apropriado;
2. procurar `Any`;
3. validar uma classe por arquivo;
4. validar nomes de módulos/pacotes;
5. detectar shadowing de bibliotecas;
6. detectar pandas linha a linha;
7. localizar hiperparâmetros hardcoded;
8. verificar pipeline de preprocessamento dentro da CV;
9. comprovar bloqueio do holdout;
10. confirmar logging do tuning no MLflow;
11. confirmar artifacts de interpretação de parâmetros para negócio;
12. executar mypy/pyright strict, ruff e pytest;
13. validar Docker Compose e provisioning Grafana.

## Saída
`production_artifacts/QA_Report.md` com PASS/FAIL por regra e evidência objetiva.
