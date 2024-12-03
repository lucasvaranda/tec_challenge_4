
# Projeto de Análise de Vídeo

Scripts Python para reconhecimento facial, detecção de poses e análise de emoções em vídeos.

## Scripts

1. **`get_faces.py`**: Extrai rostos de um vídeo para criar um dataset.
2. **`capture_poses.py`**: Usa a webcam para ajustar funções de detecção de pose.
3. **`video_analisys.py`**: Script principal que combina reconhecimento facial, poses e emoções.

## Pré-requisitos

- Python 3.7 ou superior
- Webcam (para `capture_poses.py`)
- Vídeo de entrada (para `get_faces.py` e `video_analisys.py`)

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/lucasvaranda/tec_challenge_4
   ```

## 1. Gerar Dataset de Rostos (`get_faces.py`)

**Descrição**: Extrai rostos de um vídeo para criar um dataset de reconhecimento facial.

**Execução**:
```bash
python get_faces.py
```

**Parâmetros**:
- `input_video_path`: Caminho do vídeo de entrada.
- `output_folder`: Pasta para salvar as imagens dos rostos.

Ajuste `frame_skip` e `margin` conforme necessário.

## 2. Ajustar Funções de Pose (`capture_poses.py`)

**Descrição**: Usa a webcam para testar e ajustar funções de detecção de pose.

**Execução**:
```bash
python capture_poses.py
```

**Notas**:
- Modifique as funções de pose no script para melhorar a detecção.
- Pressione `q` para sair.

## 3. Análise Principal (`video_analisys.py`)

**Descrição**: Realiza reconhecimento facial, detecção de poses, análise de emoções e geração de relatório de um vídeo.

**Pré-requisitos**:
- Coloque imagens de rostos conhecidos na pasta `images`. E redefina o nome das imagens para o nome dos individuos com 1 número no final.

**Execução**:
```bash
python video_analisys.py
```

**Parâmetros**:
- `input_video_path`: Caminho do vídeo de entrada.
- `output_video_path`: Caminho do vídeo de saída anotado.

**Resultados**:
- Vídeo Anotado: Salvo como `output_video_recognize.mp4`.
- Resumo: Estatísticas em `summary.txt`.

**Notas**:
- Desempenho: O processamento pode ser intenso; uso de GPU é recomendado.
- Ajustes: Modifique `THRESHOLD` no script para ajustar a sensibilidade do reconhecimento facial.
- Modelos: A primeira execução do DeepFace pode baixar modelos (requer internet).
