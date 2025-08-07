import streamlit as st
import assemblyai as aai
import torch
import os
import io
import zipfile
from pathlib import Path
import time
import datetime
import pandas as pd
import tempfile
from types import SimpleNamespace

# Tenta importar as bibliotecas de otimização
try:
    import whisper
except ImportError:
    st.error("Biblioteca 'openai-whisper' não encontrada. Instale com 'pip install openai-whisper'")
    st.stop()
try:
    from faster_whisper import WhisperModel
except ImportError:
    # Não para a execução, apenas desabilita a opção
    st.error("Biblioteca 'faster_whisper' não encontrada. Instale com 'pip install faster_whisper'")
    WhisperModel = None
try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    st.error("Biblioteca 'transformers' não encontrada. Instale com 'pip install transformers'")
    hf_pipeline = None


# ===================================================================
# CONFIGURAÇÃO DA PÁGINA
# ===================================================================
st.set_page_config(
    page_title="Pipeline de Transcrição v4.0",
    page_icon="⚡",
    layout="wide",
)
# ===================================================================
# ESTILIZAÇÃO PERSONALIZADA
# ===================================================================

st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .stButton > button {
        border-radius: 8px;
        padding: 10px 20px;
    }
</style>""", unsafe_allow_html=True)

# ===================================================================
# FUNÇÕES DE CACHE E CORE
# ===================================================================

@st.cache_resource
def carregar_modelo_local(_implementacao, _nome_modelo, _compute_type):
    """Carrega um modelo local (Whisper, Faster, Distil) e o mantém em cache."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with st.spinner(f"Carregando o modelo '{_nome_modelo}' via '{_implementacao}'... (Isso pode ser demorado na primeira vez)."):
        if _implementacao == "Whisper Padrão (OpenAI)":
            return whisper.load_model(_nome_modelo, device=device)
        
        if _implementacao == "Faster Whisper (Otimizado)":
            if WhisperModel is None:
                st.error("Biblioteca 'faster-whisper' não instalada. Não é possível carregar o modelo.")
                return None
            # Otimização para CPU se não houver GPU
            if device == "cpu" and _compute_type not in ["int8", "float32"]:
                _compute_type = "int8"
            return WhisperModel(_nome_modelo, device=device, compute_type=_compute_type)

        if _implementacao == "Distil-Whisper (Hugging Face)":
            if hf_pipeline is None:
                st.error("Bibliotecas 'transformers', 'sentencepiece', 'accelerate' não instaladas.")
                return None
            return hf_pipeline(
                "automatic-speech-recognition",
                model=_nome_modelo,
                device=0 if device == "cuda" else -1
            )
    return None

def extrair_audios_do_zip(zip_bytes):
    # (Sua função original - sem alterações)
    try:
        audio_files_info = []
        extensoes_audio = ['.ogg', '.mp3', '.m4a', '.wav', '.opus']
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for file_info in z.infolist():
                if not file_info.is_dir() and not file_info.filename.startswith('__MACOSX'):
                    if any(file_info.filename.lower().endswith(ext) for ext in extensoes_audio):
                        file_bytes = z.read(file_info.filename)
                        in_memory_file = io.BytesIO(file_bytes)
                        in_memory_file.name = Path(file_info.filename).name
                        audio_files_info.append({"path_completo": file_info.filename, "objeto_arquivo": in_memory_file})
        return audio_files_info
    except zipfile.BadZipFile:
        st.error("Erro: O arquivo .ZIP parece estar corrompido ou não é um ZIP válido.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao ler o ZIP: {e}")
        return None

def formatar_dialogo(utterances, com_markdown=False):
    # (Sua função original - sem alterações)
    if not utterances:
        return "Nenhuma fala detectada para formatação de diálogo."
    dialogo_formatado = []
    for utterance in utterances:
        timestamp_ms = utterance.start
        if timestamp_ms is None: timestamp_ms = 0
        segundos = int(max(0, timestamp_ms) / 1000)
        timestamp = str(datetime.timedelta(seconds=segundos))
        
        locutor = f"Locutor {utterance.speaker}" if hasattr(utterance, 'speaker') and utterance.speaker else "Fala"

        if com_markdown:
            linha_dialogo = f"**`[{timestamp}]` {locutor}:** {utterance.text.strip()}"
        else:
            linha_dialogo = f"[{timestamp}] {locutor}: {utterance.text.strip()}"
        dialogo_formatado.append(linha_dialogo)
    return "\n\n".join(dialogo_formatado)

def preparar_download(transcritos, formato_pacote, tipo_texto):
    # (Sua função original - sem alterações)
    if not transcritos: return None
    def criar_concatenado():
        conteudo = ""
        for transc in transcritos:
            conteudo += f"///// {transc['path_completo']} /////\n\n{transc[tipo_texto]}\n\n\n"
        return conteudo
    def criar_zip(incluir_concatenado=False):
        memoria_zip = io.BytesIO()
        with zipfile.ZipFile(memoria_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for transc in transcritos:
                path_txt = str(Path(transc['path_completo']).with_suffix('.txt'))
                zf.writestr(path_txt, transc[tipo_texto])
            if incluir_concatenado:
                zf.writestr("_resultado_final_concatenado.txt", criar_concatenado())
        memoria_zip.seek(0)
        return memoria_zip
    if formato_pacote == "concatenado": return {"data": criar_concatenado().encode("utf-8"), "file_name": "transcricao_concatenada.txt", "mime": "text/plain"}
    if formato_pacote == "zip_individuais": return {"data": criar_zip(False), "file_name": "transcricoes_individuais.zip", "mime": "application/zip"}
    if formato_pacote == "pacote_completo": return {"data": criar_zip(True), "file_name": "pacote_completo_transcricoes.zip", "mime": "application/zip"}
    return None

# ===================================================================
# INTERFACE DO STREAMLIT (FLUXO PRINCIPAL)
# ===================================================================

st.title("🚀 Pipeline de Transcrição v4.0 (Híbrido e Otimizado)")

if 'transcripts' not in st.session_state: st.session_state.transcripts = []

with st.expander("PASSO 1: Configurar a Análise", expanded=True):
    engine_selecionado = st.radio(
        "Escolha o motor de transcrição:",
        ("AssemblyAI (Nuvem)", "Modelos Locais"),
        horizontal=True,
        help="AssemblyAI usa a nuvem (precisa de API Key), é rápido e tem mais recursos. Modelos Locais rodam no seu computador e são gratuitos."
    )
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        if engine_selecionado == "AssemblyAI (Nuvem)":
            api_key = st.text_input("🔑 Chave da API da AssemblyAI", type="password", key="api_key_aai")
            st.session_state.api_key = api_key
            st.session_state.local_config = None
        else: # Modelos Locais
            st.session_state.api_key = None
            
            implementacao = st.selectbox(
                "⚡ Escolha a implementação Whisper:",
                ("Faster Whisper (Otimizado)", "Distil-Whisper (Hugging Face)", "Whisper Padrão (OpenAI)"),
                help="**Faster Whisper:** Mais rápido e leve. **Distil-Whisper:** Ótimo balanço de velocidade/qualidade. **Padrão:** A implementação original da OpenAI."
            )

            if implementacao == "Whisper Padrão (OpenAI)":
                model_name = st.selectbox("🧠 Modelo:", ("tiny", "base", "small", "medium", "large"))
                compute_type = "default"
            
            elif implementacao == "Faster Whisper (Otimizado)":
                model_name = st.selectbox("🧠 Modelo:", ("tiny", "base", "small", "medium", "large-v2", "large-v3"))
                compute_type = st.selectbox("⚙️ Precisão (GPU/CPU):", 
                                            ("float16", "int8_float16", "int8"),
                                            index=0 if torch.cuda.is_available() else 2,
                                            format_func=lambda x: f"{x} (Recomendado GPU)" if x != "int8" else f"{x} (Recomendado CPU)",
                                            help="float16 é mais preciso (GPU). int8 é mais leve (CPU/GPU).")

            else: # Distil-Whisper
                model_name = st.selectbox("🧠 Modelo:", (
                    "distil-whisper/distil-large-v2", 
                    "distil-whisper/distil-medium.en", 
                    "distil-whisper/distil-small.en"
                ))
                compute_type = "default"
                if not model_name.endswith(".en") and st.checkbox("Ativar modo 'long-form' (para áudios > 30s)", value=True):
                    st.session_state.distil_long_form = True
                else:
                    st.session_state.distil_long_form = False


            st.session_state.local_config = {
                "implementacao": implementacao,
                "nome_modelo": model_name,
                "compute_type": compute_type
            }
            st.caption(f"ℹ️ Rodando localmente com **{implementacao}** e modelo **{model_name}**.")


    with col2:
        locais_selecionado = (engine_selecionado == "Modelos Locais")
        st.markdown("**✨ Power-ups de IA:**")
        config_entity_detection = st.checkbox("Detectar Entidades", value=True, disabled=locais_selecionado, help="Disponível apenas com AssemblyAI.")
        if locais_selecionado:
            st.caption("Recursos como Detecção de Entidades e Diarização por Locutor (A, B, C...) não são suportados nativamente pelos modelos locais neste pipeline.")


with st.expander("PASSO 2: Enviar os Áudios", expanded=True):
    # (Seu código do Passo 2 - sem alterações)
    input_method = st.radio("Como você quer enviar os áudios?", ("Upload Direto de Arquivos", "Upload de um arquivo .ZIP"), horizontal=True)
    uploaded_audios_info = []
    if input_method == "Upload Direto de Arquivos":
        uploaded_files = st.file_uploader("Arraste e solte seus áudios aqui", type=['.ogg', '.mp3', '.m4a', '.wav', '.opus'], accept_multiple_files=True)
        if uploaded_files: uploaded_audios_info = [{"path_completo": f.name, "objeto_arquivo": f} for f in uploaded_files]
    else:
        uploaded_zip = st.file_uploader("Arraste e solte seu arquivo .ZIP aqui", type=['zip'])
        if uploaded_zip:
            uploaded_audios_info = extrair_audios_do_zip(uploaded_zip.getvalue())
            if uploaded_audios_info is not None and not uploaded_audios_info: st.warning("Atenção: Nenhum áudio compatível foi encontrado dentro do ZIP.")
            elif uploaded_audios_info: st.info(f"Encontrei e preparei {len(uploaded_audios_info)} áudios, preservando suas pastas.")

st.header("PASSO 3: Processar")

botao_desabilitado = not uploaded_audios_info
if engine_selecionado == "AssemblyAI (Nuvem)" and not st.session_state.get('api_key'):
    botao_desabilitado = True
elif engine_selecionado == "Modelos Locais" and not st.session_state.get('local_config'):
     botao_desabilitado = True


if st.button("▶️ INICIAR PIPELINE COMPLETO", type="primary", use_container_width=True, disabled=botao_desabilitado):
    st.session_state.transcripts = []
    progress_bar = st.progress(0, text="Iniciando...")

    try:
        if engine_selecionado == "AssemblyAI (Nuvem)":
            # (Seu código AssemblyAI - sem alterações)
            aai.settings.api_key = st.session_state.api_key
            config = aai.TranscriptionConfig(speaker_labels=True, entity_detection=config_entity_detection, language_code="pt", speech_model=aai.SpeechModel.best)
            transcriber = aai.Transcriber(config=config)
            
            for i, audio_info in enumerate(uploaded_audios_info):
                path_completo = audio_info["path_completo"]
                progress_bar.progress((i) / len(uploaded_audios_info), text=f"Processando com AssemblyAI: '{path_completo}' ({i+1}/{len(uploaded_audios_info)})")
                transcript = transcriber.transcribe(audio_info["objeto_arquivo"])
                if transcript.status == aai.TranscriptStatus.error:
                    st.warning(f"⚠️ Falha ao transcrever '{path_completo}': {transcript.error}. Pulando.")
                    continue
                st.session_state.transcripts.append({
                    "path_completo": path_completo, 
                    "texto_original": transcript.text or " ",
                    "texto_formatado_md": formatar_dialogo(transcript.utterances, com_markdown=True),
                    "texto_formatado_simples": formatar_dialogo(transcript.utterances, com_markdown=False),
                    "entidades": transcript.entities if config_entity_detection else []
                })

        else: # --- CAMINHO MODELOS LOCAIS (Whisper, Faster, Distil) ---
            conf = st.session_state.local_config
            modelo = carregar_modelo_local(conf['implementacao'], conf['nome_modelo'], conf['compute_type'])
            
            if modelo is None:
                st.error("Falha ao carregar o modelo local. Verifique as configurações e se as bibliotecas estão instaladas.")
                st.stop()

            for i, audio_info in enumerate(uploaded_audios_info):
                audio_file, path_completo = audio_info["objeto_arquivo"], audio_info["path_completo"]
                progress_bar.progress(i / len(uploaded_audios_info), text=f"Preparando: '{path_completo}' ({i+1}/{len(uploaded_audios_info)})")
                
                tmp_audio_path = None
                texto_final = ""
                pseudo_utterances = []

                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(path_completo).suffix) as tmp:
                        tmp.write(audio_file.getvalue())
                        tmp_audio_path = tmp.name

                    with st.spinner(f"⏳ O modelo '{conf['nome_modelo']}' está transcrevendo '{path_completo}'... Seja paciente."):
                        
                        # --- LÓGICA DE TRANSCRIÇÃO PARA CADA IMPLEMENTAÇÃO ---

                        if conf['implementacao'] == "Whisper Padrão (OpenAI)":
                            resultado = modelo.transcribe(tmp_audio_path, language='pt', fp16=torch.cuda.is_available())
                            texto_final = resultado.get('text', ' ')
                            if 'segments' in resultado:
                                for seg in resultado['segments']:
                                    pseudo_utterances.append(SimpleNamespace(start=seg['start']*1000, text=seg['text'], speaker=None))
                        
                        elif conf['implementacao'] == "Faster Whisper (Otimizado)":
                            segments, info = modelo.transcribe(tmp_audio_path, language='pt', word_timestamps=True)
                            text_parts = []
                            for seg in segments:
                                text_parts.append(seg.text)
                                pseudo_utterances.append(SimpleNamespace(start=seg.start*1000, text=seg.text.strip(), speaker=None))
                            texto_final = "".join(text_parts).strip()

                        elif conf['implementacao'] == "Distil-Whisper (Hugging Face)":
                            kwargs = {"generate_kwargs": {"language": "portuguese"}} if not conf['nome_modelo'].endswith(".en") else {}
                            resultado = modelo(
                                tmp_audio_path,
                                chunk_length_s=30 if st.session_state.get('distil_long_form', False) else 0,
                                stride_length_s=5 if st.session_state.get('distil_long_form', False) else 0,
                                return_timestamps=True,
                                **kwargs
                            )
                            texto_final = resultado['text'].strip()
                            if 'chunks' in resultado:
                                for chunk in resultado['chunks']:
                                    pseudo_utterances.append(SimpleNamespace(start=chunk['timestamp'][0]*1000, text=chunk['text'].strip(), speaker=None))
                
                finally:
                    if tmp_audio_path and os.path.exists(tmp_audio_path):
                        os.remove(tmp_audio_path)
                
                # Adicionar resultado ao session_state
                st.session_state.transcripts.append({
                    "path_completo": path_completo,
                    "texto_original": texto_final or " ",
                    "texto_formatado_md": formatar_dialogo(pseudo_utterances, com_markdown=True),
                    "texto_formatado_simples": formatar_dialogo(pseudo_utterances, com_markdown=False),
                    "entidades": []
                })

    except Exception as e:
        st.error(f"❌ ERRO GERAL NO PIPELINE: {e}")
        import traceback
        st.code(traceback.format_exc()) # Para debug
        if engine_selecionado == "AssemblyAI (Nuvem)":
            st.info("Pode ser uma chave de API inválida ou problemas de conexão.")
        else:
            st.info("Pode ser um problema com a instalação do modelo, falta de memória ou incompatibilidade de hardware.")
    finally:
        progress_bar.progress(1.0, "Pipeline finalizado!")
        if not st.session_state.transcripts: st.error("Nenhum arquivo pôde ser transcrito.")
        else: st.success(f"{len(st.session_state.transcripts)} arquivo(s) transcrito(s) com sucesso!")
        time.sleep(2); progress_bar.empty()

# ===================================================================
# PASSO 4: EXPLORAR E BAIXAR
# ===================================================================
if st.session_state.transcripts:
    st.header("PASSO 4: Explorar e Baixar os Resultados")
    # (Seu código do Passo 4 - sem alterações)
    usar_dialogo = st.toggle("Exibir/Baixar como Diálogo Formatado", value=True)
    
    for i, result in enumerate(st.session_state.transcripts):
        with st.expander(f"📄 **{result['path_completo']}**", expanded=i==0):
            texto_para_exibir = result['texto_formatado_md'] if usar_dialogo else result['texto_original']
            
            tabs = st.tabs(["Transcrição", "Entidades"])
            with tabs[0]: st.markdown(texto_para_exibir, unsafe_allow_html=True)
            with tabs[1]:
                if result.get('entidades'):
                    df_entidades = pd.DataFrame([vars(ent) for ent in result['entidades']])
                    st.dataframe(df_entidades[['entity_type', 'text', 'start', 'end']].rename(columns={'entity_type':'Tipo', 'text':'Texto', 'start':'Início (ms)', 'end':'Fim (ms)'}))
                else:
                    st.write("Detecção de entidades não ativada ou não suportada por este motor.")

    st.subheader("📦 Opções de Download")
    if usar_dialogo:
        formato_texto_download = st.radio("Formatação do texto:", ("texto_formatado_simples", "texto_formatado_md"),
                                          format_func=lambda x: "Texto Simples" if x == "texto_formatado_simples" else "Com Markdown", horizontal=True)
        tipo_texto_final = formato_texto_download
    else:
        st.info("O download será feito com o texto original (não formatado).")
        tipo_texto_final = "texto_original"
    
    formato_pacote = st.radio("Escolha o formato do pacote:", ("pacote_completo", "zip_individuais", "concatenado"),
                              format_func=lambda x: {"pacote_completo": "Pacote Completo", "zip_individuais": ".txt Individuais", "concatenado": "Arquivo Único"}[x])
    
    download_info = preparar_download(st.session_state.transcripts, formato_pacote, tipo_texto_final)
    if download_info:
        st.download_button(label="📥 BAIXAR RESULTADOS", data=download_info["data"], file_name=download_info["file_name"],
                          mime=download_info["mime"], use_container_width=True, type="primary")