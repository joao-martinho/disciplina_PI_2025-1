import numpy as np

dados_entrada = [[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]]
saidas_esperadas = [0, 0, 0, 1]

np.random.seed(42)
pesos = np.random.rand(2)
vies = np.random.rand(1)
taxa_aprendizado = 0.1

def funcao_ativacao(soma):
    return 1 if soma >= 0 else 0

max_epocas = 10
convergiu = False

for epoca in range(max_epocas):
    erros = 0

    for i in range(len(dados_entrada)):
        soma_ponderada = np.dot(dados_entrada[i], pesos) + vies
        saida_predita = funcao_ativacao(soma_ponderada)

        if saida_predita == saidas_esperadas[i]:
            continue

        erros += 1
        if saida_predita == 0:
            pesos += taxa_aprendizado * np.array(dados_entrada[i])
            vies += taxa_aprendizado * 1
        else:
            pesos -= taxa_aprendizado * np.array(dados_entrada[i])
            vies -= taxa_aprendizado * 1

    if erros == 0:
        convergiu = True
        break

print("Pesos finais:", pesos)
print("Viés final:", vies)

print("\nTeste final:")
for i in range(len(dados_entrada)):
    soma = np.dot(dados_entrada[i], pesos) + vies
    saida = funcao_ativacao(soma)
    print(f"Entrada: {dados_entrada[i]} → Saída: {saida} (Esperado: {saidas_esperadas[i]})")
