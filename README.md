# Quality Vision

Sistema de inspeção automatizada de qualidade industrial baseado em visão computacional. Classifica peças de metal fundido em tempo real, identificando defeitos de fabricação com alta precisão através de uma rede neural convolucional treinada com transfer learning.

---

## Contexto

Controle de qualidade é um dos maiores gargalos em linhas de produção industrial. Em fundições, a inspeção manual de peças é lenta, cara e sujeita à fadiga humana — um operador avaliando centenas de peças por turno inevitavelmente deixa defeitos passarem. Defeitos não detectados geram recalls, desperdício de material, paradas de linha e, em aplicações críticas como componentes hidráulicos, riscos de segurança.

A automação dessa inspeção via visão computacional resolve o problema na raiz: velocidade consistente, sem fadiga, com rastreabilidade de cada decisão. O desafio está em construir um sistema que seja preciso o suficiente para uso real, explicável o suficiente para o operador confiar, e simples o suficiente para rodar no hardware disponível na fábrica.

Quality Vision é uma resposta direta a esse problema. O sistema recebe a foto de uma peça, classifica como aprovada ou com defeito em menos de um segundo, e ainda mostra exatamente qual região da peça influenciou a decisão — sem caixa preta.

---

## Dataset

O projeto usa o [Real-Life Industrial Dataset of Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product), composto por 7.348 imagens reais de impellers (rodas de bomba d'água) capturadas em ambiente industrial.

As imagens foram obtidas em condições altamente controladas: mesma câmera, mesmo ângulo de captura, fundo branco uniforme e iluminação constante. Essa padronização é intencional no contexto industrial — linhas de inspeção automatizada operam exatamente assim, com a peça posicionada de forma fixa diante de uma câmera estacionária.

Essa característica do dataset explica diretamente a acurácia elevada alcançada no conjunto de teste. O modelo não precisa ser invariante a rotações arbitrárias, mudanças de perspectiva ou variações de iluminação severas — o ambiente de captura já elimina essas variáveis. Em implantação real, o mesmo ambiente controlado seria reproduzido na linha de produção, tornando a alta precisão sustentável.

O dataset apresenta um desequilíbrio leve entre classes (3.137 peças OK vs 4.211 com defeito), o que foi considerado na avaliação ao priorizar métricas como Recall e F1-Score além da acurácia bruta.

Uma particularidade identificada durante o desenvolvimento: o dataset original contém 144 imagens com nomes duplicados entre as pastas `train/` e `test/`, configurando data leakage. O pipeline ETL do projeto trata esse problema automaticamente, removendo do treino qualquer imagem cujo nome apareça no conjunto de teste.

---

## Arquitetura do sistema

```
quality-vision/
├── backend/
│   ├── etl/
│   │   └── preprocess.py       # Download automático e pipeline ETL
│   ├── model/
│   │   ├── model.py            # Arquitetura MobileNetV2
│   │   ├── train.py            # Treinamento com Mixed Precision
│   │   └── evaluate.py         # Avaliação no conjunto de teste
│   └── api/
│       └── main.py             # API FastAPI com GradCAM
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── InspectionPage.tsx
│       │   └── DashboardPage.tsx
│       └── hooks/
│           ├── api.ts
│           └── useTrainingLogs.ts
├── data/
│   ├── raw/                    # Dataset original (ignorado pelo git)
│   └── processed/              # Imagens pré-processadas
├── models/
│   └── checkpoints/            # Pesos salvos do modelo
└── logs/                       # Logs de treinamento (JSON + SQLite)
```

---

## Módulos

### ETL (`backend/etl/preprocess.py`)

O pipeline começa com o download automático do dataset via Kaggle API, sem nenhuma intervenção manual além de configurar as credenciais no `.env`. Na primeira execução, o script autentica, baixa e extrai os dados; nas seguintes, detecta que o dataset já existe e pula direto para o processamento.

O pré-processamento consiste em redimensionar todas as imagens para 224×224 pixels (padrão ImageNet) e convertê-las para RGB. A divisão dos dados respeita a separação original do dataset — o conjunto de teste vem exclusivamente da pasta `test/` original, e o conjunto de validação é extraído estratificado a partir do `train/` original. Isso evita o data leakage descrito acima e garante que as métricas de avaliação reflitam desempenho real.

Os arquivos são salvos com prefixo indicando a origem (`train_` ou `test_`), eliminando colisões de nome entre splits.

### Modelo (`backend/model/model.py`)

A escolha pelo MobileNetV2 pré-treinado no ImageNet foi deliberada. Redes mais pesadas como ResNet-50 ou EfficientNet-B4 entregariam ganhos marginais de precisão nesse dataset a um custo de memória e tempo de inferência proibitivos para uso em produção. O MobileNetV2 roda confortavelmente em 6GB de VRAM com batch size 32 e entrega inferência em menos de 50ms por imagem — viável para inspeção em linha.

A cabeça de classificação substitui o classificador original por uma sequência com Dropout (0.5), camada densa de 1280→256 neurônios, ReLU, Dropout adicional (0.25) e saída única com BCEWithLogitsLoss. Usar logits diretamente (sem Sigmoid na saída) é numericamente mais estável durante o treino.

### Treinamento (`backend/model/train.py`)

O treinamento segue uma estratégia de duas fases:

**Fase 1 (épocas 1–6):** O backbone está congelado. Apenas a cabeça de classificação é treinada com learning rate `1e-3`. Essa fase aproveita as features do ImageNet sem correr o risco de destruí-las com gradientes grandes logo no início.

**Fase 2 (a partir da época 7):** As últimas 5 camadas do backbone são descongeladas para fine-tuning com learning rate reduzido (`1e-4`). O modelo ajusta as representações de alto nível para as características específicas de superfícies metálicas fundidas.

Mixed Precision (FP16) está habilitado quando GPU está disponível, reduzindo o consumo de VRAM em ~40% e acelerando o treino sem perda de precisão.

A augmentação de dados durante o treino inclui flips horizontais e verticais, rotações, variações de brilho/contraste/saturação, perspectiva aleatória, blur gaussiano e random erasing. Essas transformações forçam o modelo a aprender características intrínsecas das peças e não artefatos do processo de captura, o que contribui diretamente para a generalização observada em imagens fora do dataset.

Early stopping com paciência de 7 épocas e `ReduceLROnPlateau` evitam overfitting e desperdício de tempo de computação.

### API (`backend/api/main.py`)

A API expõe dois endpoints de inferência. O `/predict` retorna a classificação e as probabilidades. O `/predict-gradcam` retorna o mesmo mais o mapa de atenção GradCAM sobreposto à imagem original, codificado em base64.

O GradCAM funciona registrando hooks na última camada convolucional do MobileNetV2 (`features[-1]`, output 1280×7×7), extraindo os gradientes da predição em relação a essa camada, ponderando as ativações pelos gradientes médios (Global Average Pooling) e fazendo upsample para 224×224. O resultado é um heatmap que indica quais regiões da imagem mais influenciaram a decisão — vermelho intenso marca alta atenção, azul marca baixa atenção.

Endpoints de monitoramento (`/metrics`, `/training-status`) são consultados pelo frontend via polling a cada 2 segundos durante o treinamento.

### Frontend (`frontend/src/`)

Interface construída em React com TypeScript, sem UI library externa — todos os estilos são inline via objetos CSS-in-JS, o que elimina dependências de build e garante comportamento previsível. As cores e tokens de design são definidos via CSS variables no `index.css`.

O design segue uma estética escura industrial: fundo `#0a0c10`, superfícies em `#111318`, acento em `#00d4aa`. A tipografia mistura Inter para texto corrido e Space Mono para dados numéricos, métricas e identificadores — uma distinção visual que separa informação humana de informação de máquina.

O `DashboardPage` consome os logs de treinamento via polling e renderiza curvas de acurácia e loss em tempo real usando Recharts. Isso permite acompanhar o progresso do treinamento sem precisar ficar olhando para o terminal.

---

## Resultados

O modelo treinado com o pipeline completo (data leakage corrigido, augmentação expandida) alcança no conjunto de teste:

| Métrica | Valor |
|---|---|
| Acurácia | ~99% |
| Precisão | ~99% |
| Recall | ~99% |
| F1-Score | ~99% |

O Recall elevado é especialmente relevante para o contexto industrial: significa que a taxa de falsos negativos (defeitos que passam como aprovados) é mínima — o erro de maior custo nesse domínio.

Em testes com imagens fora do dataset — peças fotografadas com câmeras diferentes, iluminação variada e ângulos distintos — o modelo demonstrou generalização razoável, classificando corretamente peças com defeitos visualmente similares aos do dataset de treino. A augmentação com perspectiva aleatória e blur gaussiano foi determinante para esse resultado.

---

## Como rodar

### Pré-requisitos

- Python 3.11
- Node.js 18+
- GPU NVIDIA com CUDA 11.8+ (recomendado; roda em CPU também, mais lento)
- Conta no Kaggle com API token

### Configuração

```bash
# clone o repositório
git clone https://github.com/seu-usuario/quality-vision
cd quality-vision

# crie e ative o ambiente virtual
py -3.11 -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

# instale as dependências Python
pip install -r requirements.txt

# instale PyTorch com suporte a GPU (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# confirme que a GPU foi detectada
python -c "import torch; print(torch.cuda.is_available())"

# instale as dependências do frontend
cd frontend && npm install && cd ..
```

Crie um arquivo `.env` na raiz do projeto:

```
KAGGLE_USERNAME=seu_usuario
KAGGLE_KEY=sua_chave_api
```

Para gerar sua chave: [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token.

### Executando

**Terminal 1 — ETL e treinamento**

```bash
# baixa o dataset e processa as imagens (só na primeira vez)
python backend/etl/preprocess.py

# treina o modelo (~60–90 min na RTX 3050 6GB)
python -m backend.model.train

# avalia no conjunto de teste
python -m backend.model.evaluate
```

**Terminal 2 — API**

```bash
python -m uvicorn backend.api.main:app --reload --port 8000
```

**Terminal 3 — Frontend**

```bash
cd frontend
npm run dev
```

Acesse `localhost:5173`. O dashboard de métricas pode ser aberto durante o treinamento para acompanhar as curvas em tempo real.

---

## Dependências principais

**Backend:** Python 3.11, PyTorch 2.x, torchvision, FastAPI, Pillow, scikit-learn, opencv-python-headless, kaggle, tqdm

**Frontend:** React 18, TypeScript, Vite, Recharts, lucide-react

---

## Limitações conhecidas

O modelo foi treinado em imagens capturadas em ambiente controlado. A acurácia tende a cair em cenários com iluminação muito diferente, sombras fortes, ângulos incomuns ou peças de geometria muito distinta dos impellers do dataset original. Para uso em produção com tipos de peças diferentes, o recomendado é coletar um dataset próprio e realizar fine-tuning a partir dos pesos treinados.

O endpoint `/predict-gradcam` é mais lento que o `/predict` por precisar de um forward pass com gradientes habilitados. Para inspeção em alta velocidade, use `/predict` e acione o GradCAM apenas quando quiser investigar um resultado específico.
