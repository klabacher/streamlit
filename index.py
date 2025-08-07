# app_v2.8.py

# RoadMap
# 1: Incluir Whisper como opção de modelo e deixar disponivel alguns modelos leves 
# 2: Adiconar opção de double streaming processing
# 3: Integrar langgraph com Grok para sumarização
# 4: Integrar langraph com Grok para análise de sentimentos
# 5: Adicionar Analise de performance do operador com grok api + matplotlyb
#

import streamlit as st
import assemblyai as aai
import os
import io
import zipfile
from pathlib import Path
import time
import datetime
import pandas as pd

# ===================================================================
# CONFIGURAÇÃO DA PÁGINA
# ===================================================================
st.set_page_config(
    page_title="Pipeline de Transcrição v2.8",
    page_icon="🎯",
    layout="wide",
)

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
# FUNÇÕES DO CORE
# ===================================================================

def extrair_audios_do_zip(zip_bytes):
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
    if not utterances:
        return "Nenhuma fala detectada para formatação de diálogo."
    dialogo_formatado = []
    for utterance in utterances:
        timestamp_ms = utterance.start
        if timestamp_ms is None: timestamp_ms = 0
        segundos, _ = divmod(max(0, timestamp_ms), 1000)
        timestamp = str(datetime.timedelta(seconds=segundos))
        locutor = f"Locutor {utterance.speaker}"
        if com_markdown:
            linha_dialogo = f"**`[{timestamp}]` {locutor}:** {utterance.text}"
        else:
            linha_dialogo = f"[{timestamp}] {locutor}: {utterance.text}"
        dialogo_formatado.append(linha_dialogo)
    return "\n\n".join(dialogo_formatado)

def preparar_download(transcritos, formato_pacote, tipo_texto):
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

st.title("🎯 Pipeline de Transcrição v2.8 (Final)")

if 'transcripts' not in st.session_state: st.session_state.transcripts = []

with st.expander("PASSO 1: Configurar a Análise", expanded=True):
    col1, col2 = st.columns([2, 3])
    with col1:
        api_key = st.text_input("🔑 Chave da API da AssemblyAI", type="password")
    with col2:
        st.markdown("**✨ Power-ups de IA:**")
        config_entity_detection = st.checkbox("Detectar Entidades", value=True)

with st.expander("PASSO 2: Enviar os Áudios", expanded=True):
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

botao_desabilitado = (not api_key or not uploaded_audios_info)
if input_method == "Upload de um arquivo .ZIP":
    if 'uploaded_zip' in locals() and uploaded_zip and (uploaded_audios_info is None or not uploaded_audios_info): botao_desabilitado = True

if st.button("▶️ INICIAR PIPELINE COMPLETO", type="primary", use_container_width=True, disabled=botao_desabilitado):
    st.session_state.transcripts = []
    aai.settings.api_key = api_key
    
    config = aai.TranscriptionConfig(speaker_labels=True, entity_detection=config_entity_detection, language_code="pt", speech_model=aai.SpeechModel.best)
    
    transcriber = aai.Transcriber(config=config)
    progress_bar = st.progress(0, text="Iniciando...")
    try:
        for i, audio_info in enumerate(uploaded_audios_info):
            audio_file, path_completo = audio_info["objeto_arquivo"], audio_info["path_completo"]
            progress_bar.progress(i / len(uploaded_audios_info), text=f"Processando: '{path_completo}' ({i+1}/{len(uploaded_audios_info)})")
            try:
                transcript = transcriber.transcribe(audio_file)
                if transcript.status == aai.TranscriptStatus.error: st.warning(f"⚠️ Falha ao transcrever '{path_completo}': {transcript.error}. Pulando.")
                else:
                    st.session_state.transcripts.append({
                        "path_completo": path_completo, "texto_original": transcript.text or " ",
                        "texto_formatado_md": formatar_dialogo(transcript.utterances, com_markdown=True),
                        "texto_formatado_simples": formatar_dialogo(transcript.utterances, com_markdown=False),
                        "entidades": transcript.entities
                    })
            except Exception as e: st.warning(f"⚠️ Erro no arquivo '{path_completo}': {e}. Pulando.")
    except Exception as e:
        st.error(f"❌ ERRO GERAL NO PIPELINE: {e}")
        st.info("Pode ser uma chave de API inválida ou problemas de conexão. Verifique e tente novamente.")
    finally:
        progress_bar.progress(1.0, "Pipeline finalizado!")
        if not st.session_state.transcripts: st.error("Nenhum arquivo pôde ser transcrito.")
        else: st.success(f"{len(st.session_state.transcripts)} arquivo(s) transcrito(s) com sucesso!")
        time.sleep(2); progress_bar.empty()

if st.session_state.transcripts:
    st.header("PASSO 4: Explorar e Baixar os Resultados")
    usar_dialogo = st.toggle("Exibir/Baixar como Diálogo Formatado", value=True)
    for i, result in enumerate(st.session_state.transcripts):
        with st.expander(f"📄 **{result['path_completo']}**", expanded=i==0):
            texto_para_exibir = result['texto_formatado_md'] if usar_dialogo else result['texto_original']
            
            tabs = st.tabs(["Transcrição", "Entidades"])
            with tabs[0]: st.markdown(texto_para_exibir, unsafe_allow_html=True)
            with tabs[1]:
                if result['entidades']:
                    lista_de_entidades = [
                        {"Tipo": ent.entity_type, "Texto": ent.text, "Início (ms)": ent.start, "Fim (ms)": ent.end}
                        for ent in result['entidades']
                    ]
                    df_entidades = pd.DataFrame(lista_de_entidades)
                    st.dataframe(df_entidades)
                else:
                    st.write("Detecção de entidades não ativada ou nenhuma entidade encontrada.")

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