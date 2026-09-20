# Spec — Regras de negócio da imobiliária

Este arquivo é deliberadamente separado das specs de ML.

## Configuráveis
- `desconto_maximo_percentual`
- `comissao_percentual`
- `custo_fixo_transacao`
- `margem_seguranca_percentual`
- `limite_alerta_subprecificacao`
- `limite_alerta_sobreprecificacao`

## Métricas por imóvel
1. preço estimado;
2. preço informado pelo corretor;
3. diferença absoluta;
4. diferença percentual;
5. preço estimado por m²;
6. preço informado por m²;
7. desconto solicitado;
8. valor nominal do desconto;
9. preço após desconto;
10. desconto máximo configurado;
11. margem entre preço estimado e preço pós-desconto;
12. comissão estimada;
13. receita líquida estimada da imobiliária;
14. intervalo de preço P10/P50/P90 quando disponível;
15. margem até limite inferior de incerteza;
16. índice de atratividade do anúncio;
17. gap para mediana da zona;
18. gap para mediana do bairro, se houver;
19. prêmio/desconto por vaga;
20. prêmio/desconto por quarto;
21. prêmio/desconto por banheiro;
22. percentual de erro histórico da zona;
23. confiança operacional baseada em cobertura de dados;
24. taxa de categorias desconhecidas;
25. score de completude cadastral;
26. distância percentual para preço-alvo do vendedor;
27. margem para negociação;
28. preço recomendado de anúncio (regra configurável);
29. preço piso interno (regra configurável);
30. alerta de possível sub/sobreprecificação.

## Exemplo de “desconto seguro”
Uma regra prudente e configurável:
`desconto_seguro_pct = min(desconto_maximo_negocio, margem_incerteza_pct * fator_prudencia)`

Onde a margem de incerteza pode vir de quantis/intervalo conformal/erro histórico do segmento.

Nunca apresentar como garantia. É uma política de negociação interna.

## Regras adicionais para API
- `Preco_Informado` é contexto comercial e nunca feature de treinamento.
- Gerar alerta de sobreprecificação quando `Preco_Informado` estiver acima do preço estimado além do limite configurável.
- Gerar alerta de subprecificação quando estiver abaixo além do limite configurável.
- Se houver intervalo calibrado P10/P90, preferir a faixa de incerteza como apoio ao alerta em vez de um limite fixo simples.
- O desconto solicitado é validado antes do cálculo das métricas.
- Toda previsão recebe um `id_predicao` para permitir reconciliação com a venda real futura.
