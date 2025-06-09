# Resumos

## Artigo 1: Alagamento do rio Xopotó

O objetivo deste artigo é processar imagens obtidas por satélites para recolher características do terreno na bacia do rio Xopotó, e, com elas, compreender as
ocasiões de erosão e alagamento nessa região.

### Características do terreno

- **Área urbana (4,25%)**: 
  - Pouca infiltração no solo devido ao uso de materiais artificiais como concreto, asfalto, calçadas, etc.
  - O escoamento é apenas superficial.
  - Muitas áreas possuem sistemas de esgoto ineficientes.
  - Moradias em regiões de inundação frequente.

- **Pastagem (72,37%)**: 
  - Menor capacidade de infiltração.
  - Maior chance de alagamento.

- **Matas (17,08%)**: 
  - Alta porosidade e grande capacidade de infiltração do solo.
  - Baixa chance de alagamento.

- **Solo exposto (5,70%)**: 
  - Grande parte é de "mares de morros".
  - Alta taxa de erosão.
  - Escoamento superficial.
  - Baixa infiltração e maior chance de alagamento.

### Conclusões

A respeito da BHRX, ela não está particularmente sujeita a enchentes em épocas de cheia por causa do seu formato (mais alongada que circular); O formato retilíneo do rio favorece o arraste de sedimentos (pista falsa?). A capacidade de drenagem da bacia, no entanto, é alta, o que possibilita enchentes em épocas de vazão (?). A distribuição dos tipos de solo (maioria de pastagem) desfavorece a infiltração e favorece o alagamento. O solo exposto favorece a erosão, e, apesar de ser pequeno, é vizinho das áreas urbanas. Como resultado, as enchentes e desmoronamentos são um problema social grave.

## Artigo 2: Fotointerpretação em rodovias

O objetivo é usar técnicas de fotointerpretação (?) em imagens colhidas remotamente (usando drones) para colher informações a respeito do solo em áreas de rodovia para ajudar a prever onde são necessárias obras de engenharia para prevenção de desastres. Basicamente, o artigo é parecido com o anterior, exceto por ter um foco maior na prevenção e e nas rodovias, e não em uma região específica.

### Primeiro caso: Minas Gerais

A partir de uma voto aérea de uma estrada, observou-se uma "cicatriz" (trecho de solo arenoso causado pela erosão) em um morrinho à beira da estrada (imagem MG01). Depois, em outra foto, tirada 10 meses depois, a cicatriz tinha aumentado consideravelmente (imagem MG02). A partir disto, observou-se a necessidade de intervenção humana para prevenir desastres naturais (como desmoronamento?).

### Segundo caso: Rio Grande do Sul

A mesma coisa, mas em outra estrada (imagem RS01).

# TODO

## Prompt:

Eescreva um programa em Python que, a partir de uma foto tirada por satélite, reconheça padrões e características do terreno que estão mais propensas a desmoronamento. Aqui estão as regras de negócio:

- **Área urbana (4,25%)**: 
  - Pouca infiltração no solo devido ao uso de materiais artificiais como concreto, asfalto, calçadas, etc.
  - O escoamento é apenas superficial.
  - Muitas áreas possuem sistemas de esgoto ineficientes.
  - Moradias em regiões de inundação frequente.

- **Pastagem (72,37%)**: 
  - Menor capacidade de infiltração.
  - Maior chance de alagamento.

- **Matas (17,08%)**: 
  - Alta porosidade e grande capacidade de infiltração do solo.
  - Baixa chance de alagamento.

- **Solo exposto (5,70%)**: 
  - Grande parte é de "mares de morros".
  - Alta taxa de erosão.
  - Escoamento superficial.
  - Baixa infiltração e maior chance de alagamento.

Baseando-se nisso, o programa recebe a imagem através de um caminho no sistema operacional, processa-a e gera a mesma imagem, porém processada. O processamento consiste em reconhecer os quatro tipos de terreno acima e colorir regiões das imagens (área urbana: azul; pastagem: amarelo; matas: verde; solo exposto: vermelho). A imgem gerada deve se chamar nome_da_imagem_original_processada.png.
