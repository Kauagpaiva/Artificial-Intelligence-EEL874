import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

## Meu objetivo com esse código é desenvolver um suporte para o controle de vôo de um drone, impedindo que ele caia no chão com muita força ou voe muito alto

# Criando as variáveis do problema
Altura = ctrl.Antecedent(np.arange(0, 201, 1), 'Centímetros')
Queda = ctrl.Antecedent(np.arange(0, 101, 1), 'cm/s')
Comando = ctrl.Antecedent(np.arange(-10, 11, 1), 'Posição')
ForçaMotores = ctrl.Consequent(np.arange(0, 101, 1), 'RPM')

# Criando as funções de pertinência para a altura
Altura['Baixa'] = fuzz.trapmf(Altura.universe, [0, 0, 3, 40])
Altura['Media'] = fuzz.trapmf(Altura.universe, [20, 40, 100, 150])
Altura['Alta'] = fuzz.trapmf(Altura.universe, [100, 150, 200, 200])

# Criando as funções de pertinência para a velocidade de queda
Queda['Lenta'] = fuzz.trapmf(Queda.universe, [0, 0, 5, 20])
Queda['Média'] = fuzz.trapmf(Queda.universe, [5, 20, 33, 50])
Queda['Rápida'] = fuzz.trapmf(Queda.universe, [33, 50, 100, 100])

# Criando as funções de pertinência para o comando vindo do controle
Comando['Descer'] = fuzz.trapmf(Comando.universe, [-10, -10, -5, 0])
Comando['Manter'] = fuzz.trimf(Comando.universe, [-5, 0, 5])
Comando['Subir'] = fuzz.trapmf(Comando.universe, [0, 5, 10, 10])

ForçaMotores['Baixa'] = fuzz.trapmf(ForçaMotores.universe, [0, 0, 23, 33])
ForçaMotores['Média'] = fuzz.trapmf(ForçaMotores.universe, [23, 43, 56, 66])
ForçaMotores['Alta'] = fuzz.trapmf(ForçaMotores.universe, [56, 76, 100, 100])

# Visualizando as funções de pertinência para cada variável
Altura.view()
Queda.view()
Comando.view()
ForçaMotores.view()

# Base de Conhecimento/Regras
rule1 = ctrl.Rule(Comando['Descer'] | Altura['Alta'], ForçaMotores['Baixa'])
rule2 = ctrl.Rule((Comando['Subir'] & (Altura['Baixa'] | Altura['Media'])) | (Altura['Baixa'] & Queda['Rápida']), ForçaMotores['Alta']) 
rule3 = ctrl.Rule(Comando['Manter'] | (Altura['Media'] & Queda['Rápida']), ForçaMotores['Média'])

# Sistema Fuzzy e Simulação
ForçaMotores_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
ForçaMotores_simulador = ctrl.ControlSystemSimulation(ForçaMotores_ctrl)

Velocidade = float(input('Velocidade medida pelo sensor: '))
ForçaMotores_simulador.input["cm/s"] = Velocidade

Controle = float(input('Posição do controle: '))
ForçaMotores_simulador.input["Posição"] = Controle

AlturaMedida = float(input('Altura medida pelo sensor: '))
ForçaMotores_simulador.input["Centímetros"] = AlturaMedida

# Computando o resultado (Inferência Fuzzy + Defuzzificação)
ForçaMotores_simulador.compute()
print('A potência dos motores é de %d RPM' % round(ForçaMotores_simulador.output['RPM']))

# Visualizando as regiões
Altura.view(sim=ForçaMotores_simulador)
Queda.view(sim=ForçaMotores_simulador)
Comando.view(sim=ForçaMotores_simulador)
ForçaMotores.view(sim=ForçaMotores_simulador)
