# Spec — Equações de Predição dos Modelos

## Objetivo
Exibir no relatório a função de predição real de cada família. O termo “equação da reta” aplica-se estritamente a modelos lineares; para modelos não lineares deve ser mostrada sua função matemática ou representação equivalente.

## 1. LinearRegression
\[
\hat{y}=\beta_0+\sum_{j=1}^{p}\beta_j x_j
\]

Equação expandida obrigatória após treino:
```text
Valor_estimado = intercepto
               + b1*(num__Quartos)
               + b2*(num__Banheiros)
               + b3*(num__Vagas_Garagem)
               + b4*(num__Metragem)
               + b5*(cat__Zona_<zona1>) + ...
```

## 2. Ridge
Predição:
\[
\hat{y}=\beta_0+\sum_j\beta_jx_j
\]
Treinamento:
\[
\min_\beta \sum_i(y_i-\hat{y}_i)^2+\alpha\sum_j\beta_j^2
\]

## 3. Lasso
Predição linear, com treinamento:
\[
\min_\beta \sum_i(y_i-\hat{y}_i)^2+\alpha\sum_j|\beta_j|
\]

## 4. ElasticNet
\[
\min_\beta \frac{1}{2n}\|y-X\beta\|_2^2+\alpha\,l1\_ratio\|\beta\|_1+\frac{\alpha(1-l1\_ratio)}{2}\|\beta\|_2^2
\]
Predição continua sendo `intercept + Xβ`.

## 5. DecisionTreeRegressor
Não existe uma única reta. A predição é constante por folha:
\[
\hat{y}(x)=\sum_{m=1}^{M} c_m\,I(x\in R_m)
\]
No relatório, exportar regras das folhas mais relevantes e profundidade da árvore.

## 6. RandomForestRegressor
\[
\hat{y}(x)=\frac{1}{B}\sum_{b=1}^{B}T_b(x)
\]
Mostrar número de árvores, profundidade efetiva, importâncias e, opcionalmente, regras de árvores representativas.

## 7. GradientBoostingRegressor
\[
F_M(x)=F_0(x)+\sum_{m=1}^{M}\eta h_m(x)
\]
Mostrar `learning_rate`, `n_estimators`, profundidade e importâncias.

## 8. HistGradientBoostingRegressor
Também é um modelo aditivo de árvores:
\[
F_M(x)=F_0(x)+\sum_{m=1}^{M}\eta h_m(x)
\]
com discretização dos atributos em bins. Mostrar bins/leaf nodes/iterations disponíveis na API do estimador.

## 9. SVR
Para kernel:
\[
f(x)=\sum_{i=1}^{n}(\alpha_i-\alpha_i^*)K(x_i,x)+b
\]
Para kernel linear, também exportar `coef_` quando disponível e apresentar a expressão linear no espaço transformado.

## 10. KNeighborsRegressor
Uniforme:
\[
\hat{y}(x)=\frac{1}{k}\sum_{i\in N_k(x)}y_i
\]
Ponderado por distância:
\[
\hat{y}(x)=\frac{\sum_{i\in N_k(x)}w_i y_i}{\sum_{i\in N_k(x)}w_i}
\]

## Requisito de relatório
Para cada modelo treinado registrar:
- fórmula da família;
- hiperparâmetros finais;
- parâmetros aprendidos aplicáveis;
- equação expandida quando linear;
- interpretação de negócio;
- limitações da interpretação.
