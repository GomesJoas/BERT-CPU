import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="🧠 MLP Experimental Study",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 MLP Experimental Study - Adult Dataset")
st.caption("Relatório Científico Automatizado de Ablação e Análise Avançada de Hiperparâmetros")

try:
    df_raw = pd.read_csv("resultados.csv")
    history_raw = pd.read_csv("historico.csv")
except FileNotFoundError:
    st.error(
        "Arquivos de resultados não encontrados!\n\n"
        "Certifique-se de que 'resultados.csv' e 'historico.csv' estão no mesmo diretório do script."
    )
    st.stop()

df_raw["activation"] = df_raw["activation"].astype(str).str.lower()
df_raw["hidden"] = df_raw["hidden"].astype(int)
df_raw["lr"] = df_raw["lr"].astype(float)

history_raw["activation"] = history_raw["activation"].astype(str).str.lower()
history_raw["hidden"] = history_raw["hidden"].astype(int)
history_raw["lr"] = history_raw["lr"].astype(float)

df_raw["acc_por_gflop"] = df_raw["test_acc"] / df_raw["gflops"]

def calcular_parametros(hidden_size, input_dim=108, output_dim=2):
    return (input_dim * hidden_size) + hidden_size + (hidden_size * output_dim) + output_dim

df_raw["parameters"] = df_raw["hidden"].apply(calcular_parametros)

def render_metrics_and_scatter(df_sub, col_alvo, label_alvo):
    m_acc = df_sub.loc[df_sub["test_acc"].idxmax()]
    m_cost = df_sub.loc[df_sub["gflops"].idxmin()]
    m_eff = df_sub.loc[df_sub["acc_por_gflop"].idxmax()]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Melhor Accuracy", str(m_acc[col_alvo]).upper(), f"Test Acc: {m_acc['test_acc']:.4f}")
    c2.metric("⚡ Menor Custo", str(m_cost[col_alvo]).upper(), f"{m_cost['gflops']:.2f} GFLOPs", delta_color="inverse")
    c3.metric("🎯 Melhor Eficiência", str(m_eff[col_alvo]).upper(), f"{m_eff['acc_por_gflop']:.4f} Acc/GFLOP")
    
    st.divider()

tab_act, tab_hidden, tab_lr = st.tabs(
    [
        "✨ Activation Functions",
        "📐 Hidden Sizes",
        "🚀 Learning Rates"
    ]
)

with tab_act:
    st.header("✨ Experimento 1: Funções de Ativação")
    st.subheader("Pergunta: *'Como diferentes funções de ativação influenciam o desempenho da MLP?'*")
    st.write("Filtro do Experimento: `Hidden Size = 64` e `Learning Rate = 0.01`")
    
    # Filtragem Estrita
    df_act = df_raw[(df_raw["activation"].isin(["relu", "gelu", "tanh"])) & (df_raw["hidden"] == 64) & (df_raw["lr"] == 0.01)].copy()
    hist_act = history_raw[(history_raw["activation"].isin(["relu", "gelu", "tanh"])) & (history_raw["hidden"] == 64) & (history_raw["lr"] == 0.01)].copy()
    
    if df_act.empty:
        st.warning("Dados não encontrados para o filtro de Funções de Ativação.")
    else:
        render_metrics_and_scatter(df_act, "activation", "Função de Ativação")
        
        st.write("### Tabela de Resultados Consolidados")
        opcoes_tempo = [c for c in ["total_time", "time", "training_time"] if c in df_act.columns]
        col_tabela_act = ["activation", "train_acc", "val_acc", "test_acc", "gflops", "acc_por_gflop"] + opcoes_tempo
        
        nomes_colunas = {"activation": "Ativação", "acc_por_gflop": "Acc/GFLOP"}
        if opcoes_tempo:
            nomes_colunas[opcoes_tempo[0]] = "Tempo Total"
            
        st.dataframe(
            df_act[col_tabela_act].rename(columns=nomes_colunas),
            use_container_width=True, hide_index=True
        )
        
        st.markdown("""
        > **💡 Detalhamento dos Indicadores:**
        * **Acurácia (Train vs Val vs Test):** Permite monitorar o *overfitting* (quando a acurácia de treino cresce, mas a de validação/teste cai).
        * **GFLOPs:** Unidade de medida de bilhões de operações de ponto flutuante. Quantifica o custo computacional puro do *forward pass*.
        * **Acc/GFLOP (Eficiência):** Indica quanta acurácia o modelo entrega para cada unidade de processamento gasta. Fundamental para cenários de produção.
        """)
        
        st.divider()

        g1, g2 = st.columns(2)
        with g1:
            fig1 = px.bar(df_act, x="activation", y="test_acc", color="activation", text="test_acc",
                          title="Accuracy por Ativação", labels={"activation": "Ativação", "test_acc": "Test Accuracy"})
            fig1.update_traces(texttemplate="%{text:.4f}", textposition="auto")
            st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown("""
            **Análise de Desempenho:**
            A função que obtiver a maior barra vertical possui melhor capacidade de generalização para os dados socioeconômicos do Adult Dataset.
            * **ReLU:** Introduz não-linearidade simples cortando valores negativos a zero ($f(x) = \max(0, x)$).
            * **GELU:** Modula os valores ponderando-os pela sua probabilidade gaussiana, comum em arquiteturas estado da arte como Transformers.
            * **Tanh:** Centrada no zero, mas suscetível à saturação de gradiente em redes muito profundas.
            """)
            
        with g2:
            fig3 = px.scatter(df_act, x="gflops", y="test_acc", color="activation", text="activation", size="acc_por_gflop",
                              title="Accuracy vs Custo Computacional (Tamanho = Eficiência)",
                              labels={"gflops": "GFLOPs (Menor é Melhor)", "test_acc": "Test Accuracy (Maior é Melhor)"})
            fig3.update_traces(textposition="top center")
            st.plotly_chart(fig3, use_container_width=True)
            
            st.markdown("""
            **Análise do Espaço de Design:**
            O cenário ideal localiza-se no **canto superior esquerdo** (máxima acurácia com mínimo custo computational). 
            O tamanho das esferas representa a métrica de retorno computacional; bolhas maiores simbolizam arquiteturas mais viáveis para implantação em dispositivos com hardware limitado.
            """)
            
        st.divider()

        st.write("### Curvas de Aprendizado (Convergência e Estabilidade)")
        fig2 = px.line(hist_act, x="epoch", y="train_loss", color="activation", markers=True,
                      title="Loss de Treinamento por Época e Função de Ativação", labels={"epoch": "Época", "train_loss": "Train Loss"})
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        **Análise de Dinâmica de Otimização:**
        A velocidade com que a curva de perda decresce indica o quão amigável a superfície de perda se torna sob o efeito de cada função. 
        Curvas que decrescem de forma suave demonstram gradientes bem comportados. Ruídos excessivos ou estagnação precoce (platôs) servem de aviso para problemas de representatividade matemática da função escolhida.
        """)
        
        st.divider()

        st.write("### Comparação contra Baseline (Referência: ReLU)")
        try:
            baseline_val = df_act[df_act["activation"] == "relu"].iloc[0]
            df_baseline = df_act[df_act["activation"] != "relu"].copy()
            df_baseline["Δ Accuracy"] = ((df_baseline["test_acc"] - baseline_val["test_acc"]) / baseline_val["test_acc"]) * 100
            df_baseline["Δ FLOPs"] = ((df_baseline["gflops"] - baseline_val["gflops"]) / baseline_val["gflops"]) * 100
            
            df_baseline["Δ Accuracy"] = df_baseline["Δ Accuracy"].map(lambda x: f"{x:+.2f}%")
            df_baseline["Δ FLOPs"] = df_baseline["Δ FLOPs"].map(lambda x: f"{x:+.2f}%")
            
            st.dataframe(df_baseline[["activation", "Δ Accuracy", "Δ FLOPs"]].rename(columns={"activation": "Modelo"}), 
                         use_container_width=True, hide_index=True)
            
            st.markdown("""
            **Interpretação Científica do Impacto Relativo:**
            Esta tabela quantifica o custo-benefício de substituir a ReLU. Um valor positivo em **Δ Accuracy** significa ganho de precisão, enquanto um valor positivo em **Δ FLOPs** aponta o preço cobrado em processamento extra. Mudanças ideais trazem acurácia positiva com FLOPs negativos ou neutros.
            """)
        except IndexError:
            st.info("Não foi possível calcular o baseline completo. Certifique-se de possuir a configuração 'relu' ativa.")

with tab_hidden:
    st.header("📐 Experimento 2: Tamanhos da Camada Oculta")
    st.subheader("Pergunta: *'Como o tamanho da camada escondida influencia a capacidade da rede?'*")
    st.write("Filtro do Experimento: `Activation = relu` e `Learning Rate = 0.01`")
    
    df_hid = df_raw[(df_raw["activation"] == "relu") & (df_raw["hidden"].isin([32, 64, 128, 256])) & (df_raw["lr"] == 0.01)].sort_values("hidden").copy()
    hist_hid = history_raw[(history_raw["activation"] == "relu") & (history_raw["hidden"].isin([32, 64, 128, 256])) & (history_raw["lr"] == 0.01)].copy()
    
    if df_hid.empty:
        st.warning("Dados não encontrados para o filtro de Tamanhos Ocultos.")
    else:
        df_hid["hidden_str"] = df_hid["hidden"].astype(str)
        hist_hid["hidden_str"] = hist_hid["hidden"].astype(str)
        
        m_acc_h = df_hid.loc[df_hid["test_acc"].idxmax()]
        m_cost_h = df_hid.loc[df_hid["gflops"].idxmin()]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Melhor Accuracy", f"Hidden: {m_acc_h['hidden']}", f"Test Acc: {m_acc_h['test_acc']:.4f}")
        c2.metric("⚡ Menor Custo", f"Hidden: {m_cost_h['hidden']}", f"{m_cost_h['gflops']:.2f} GFLOPs", delta_color="inverse")
        c3.metric("🧠 Total de Parâmetros (Máx)", f"{df_hid['parameters'].max():,}", "Config: 256 Neurônios")
        
        st.divider()
        
        st.write("### Tabela de Capacidade e Volumetria")
        col_tabela_hid = ["hidden", "parameters", "train_acc", "val_acc", "test_acc", "gflops"]
        st.dataframe(df_hid[col_tabela_hid].rename(columns={"hidden": "Hidden Neurons", "parameters": "Parameters"}), 
                     use_container_width=True, hide_index=True)
        
        st.markdown("""
        > **💡 Teoria de Capacidade Arquitetural:**
        * **Parameters:** Representa o total de pesos estruturais e vieses ($W$ e $b$) que o otimizador tenta ajustar. Cresce de forma linear em relação à dimensão dos dados de entrada multiplicado pelo número de neurônios ocultos.
        * **Escalonamento:** Redes muito pequenas sofrem de *underfitting* (falta de capacidade representativa). Redes superdimensionadas memorizam ruídos do treino, degradando as métricas de teste.
        """)
        
        st.divider()

        g3, g4 = st.columns(2)
        with g3:
            fig_h1 = px.line(df_hid, x="hidden", y="test_acc", markers=True, text="test_acc",
                             title="Hidden Size × Accuracy", labels={"hidden": "Neurônios Ocultos", "test_acc": "Test Accuracy"})
            fig_h1.update_traces(textposition="top center", texttemplate="%{text:.4f}")
            st.plotly_chart(fig_h1, use_container_width=True)
            
            st.markdown("""
            **Análise do Limite de Escalonamento:**
            Observe o comportamento geométrico da curva. Se o gráfico desenhar um formato de 'U invertido', identificamos o ponto ideal de inflexão. A partir deste cume, aumentar a largura da rede adiciona complexidade desnecessária sem ganho real de inteligência preditiva.
            """)
            
        with g4:
            fig_h2 = px.bar(df_hid, x="hidden_str", y="gflops", text="gflops", color="hidden_str",
                            title="Hidden Size × GFLOPs", labels={"hidden_str": "Número de Neurônios", "gflops": "GFLOPs"})
            fig_h2.update_traces(texttemplate="%{text:.2f}", textposition="auto")
            st.plotly_chart(fig_h2, use_container_width=True)
            
            st.markdown("""
            **Análise de Custo de Computação:**
            Este gráfico ilustra como a demanda computacional responde à expansão da camada. Nas MLPs, a complexidade computacional cresce de forma diretamente proporcional à largura da camada oculta, gerando saltos nítidos de processamento a cada duplicação de neurônios.
            """)
            
        st.divider()

        g5, g6 = st.columns(2)
        with g5:
            fig_h3 = px.scatter(df_hid, x="parameters", y="test_acc", text="hidden", size="gflops",
                                title="Quantidade de Parâmetros × Accuracy (Tamanho = Custo)",
                                labels={"parameters": "Quantidade de Parâmetros", "test_acc": "Accuracy"})
            fig_h3.update_traces(textposition="top center")
            st.plotly_chart(fig_h3, use_container_width=True)
            
            st.markdown("""
            **Análise do Custo por Parâmetro:**
            Este mapeamento em formato de bolhas expõe o preço computacional exato pago por cada incremento estrutural. Ajuda a definir de forma empírica o limite onde adicionar mais equações internas deixa de fazer sentido prático para o projeto.
            """)
            
        with g6:
            fig_h4 = px.line(hist_hid, x="epoch", y="train_loss", color="hidden_str",
                             title="Curvas de Loss por Capacidade (Hidden Size)", labels={"epoch": "Épocas", "train_loss": "Train Loss", "hidden_str": "Hidden"})
            st.plotly_chart(fig_h4, use_container_width=True)
            
            st.markdown("""
            **Análise de Velocidade de Ajuste:**
            Modelos mais largos (ex: 256) tendem a despencar a curva de Loss de forma mais agressiva nas primeiras épocas por possuírem maior quantidade de caminhos matemáticos de otimização disponíveis. Monitore se essa descida rápida não resulta em convergência precoce para mínimos locais ruins.
            """)
            
        st.divider()

        st.subheader("📋 Conclusão Automática Baseada nos Dados")
        try:
            acc_32 = df_hid[df_hid["hidden"] == 32]["test_acc"].values[0]
            acc_128 = df_hid[df_hid["hidden"] == 128]["test_acc"].values[0]
            gain_32_128 = ((acc_128 - acc_32) / acc_32) * 100
            
            flops_128 = df_hid[df_hid["hidden"] == 128]["gflops"].values[0]
            flops_256 = df_hid[df_hid["hidden"] == 256]["gflops"].values[0]
            cost_increase_256 = ((flops_256 - flops_128) / flops_128) * 100
            
            st.info(f"""
            * **Análise de Escalaridade:** O aumento de 32 para 128 neurônios alterou a acurácia de teste em **{gain_32_128:+.3f}%**.
            * **Retornos Decrescentes:** O modelo com **256 neurônios** apresentou um aumento de custo computacional de **{cost_increase_256:+.1f}%** em relação ao modelo de 128. Avalie se a variação de desempenho compensa a alocação extra em produção.
            """)
        except IndexError:
            st.info("Dados específicos de hidden size (32, 128 ou 256) incompletos no dataset para computação do texto de conclusão automática.")

with tab_lr:
    st.header("🚀 Experimento 3: Taxas de Aprendizado")
    st.subheader("Pergunta: *'Como a taxa de aprendizado afeta a convergência da rede?'*")
    st.write("Filtro do Experimento: `Activation = relu` e `Hidden Size = 64`")
    
    # Filtragem Estrita
    df_lr = df_raw[(df_raw["activation"] == "relu") & (df_raw["hidden"] == 64) & (df_raw["lr"].isin([0.001, 0.005, 0.01, 0.05]))].sort_values("lr").copy()
    hist_lr = history_raw[(history_raw["activation"] == "relu") & (history_raw["hidden"] == 64) & (history_raw["lr"].isin([0.001, 0.005, 0.01, 0.05]))].copy()
    
    if df_lr.empty:
        st.warning("Dados não encontrados para o filtro de Taxas de Aprendizado.")
    else:
        df_lr["lr_str"] = df_lr["lr"].astype(str)
        hist_lr["lr_str"] = hist_lr["lr"].astype(str)
        
        st.write("### Tabela de Otimização Dinâmica")
        opcoes_tempo_lr = [c for c in ["total_time", "time", "training_time"] if c in df_lr.columns]
        col_tabela_lr = ["lr", "train_acc", "val_acc", "test_acc", "gflops"] + opcoes_tempo_lr
        
        nomes_colunas_lr = {"lr": "LR"}
        if opcoes_tempo_lr:
            nomes_colunas_lr[opcoes_tempo_lr[0]] = "Tempo Total"

        st.dataframe(
            df_lr[col_tabela_lr].rename(columns=nomes_colunas_lr), 
            use_container_width=True, hide_index=True
        )
        
        st.markdown("""
        > **💡 Teoria do Hiperparâmetro Mais Crítico:**
        * **Learning Rate ($\eta$):** Controla o tamanho do passo que o algoritmo dá em direção ao mínimo global da função de perda a cada atualização de pesos.
        * **O Dilema:** Valores muito grandes causam divergência matemática (o modelo salta além do mínimo). Valores ínfimos travam o modelo em um treinamento infinito que consome tempo e recursos sem evoluir.
        """)
        
        st.divider()

        g7, g8 = st.columns(2)
        with g7:
            fig_l1 = px.bar(df_lr, x="lr_str", y="test_acc", color="lr_str", text="test_acc",
                            title="Learning Rate × Accuracy", labels={"lr_str": "Taxa de Aprendizado", "test_acc": "Test Accuracy"})
            fig_l1.update_traces(texttemplate="%{text:.4f}", textposition="auto")
            st.plotly_chart(fig_l1, use_container_width=True)
            
            st.markdown("""
            **Análise de Estabilidade da Taxa:**
            Indica de forma clara qual magnitude de passo gerou a melhor sintonia fina nos pesos para classificar as faixas de renda do dataset. Quedas acentuadas de acurácia em LRs maiores evidenciam instabilidade no processo de descida de gradiente.
            """)
            
        with g8:
            fig_l2 = px.line(hist_lr, x="epoch", y="train_loss", color="lr_str",
                             title="Curva de Loss por Taxa de Otimização", labels={"epoch": "Épocas", "train_loss": "Train Loss", "lr_str": "LR"})
            st.plotly_chart(fig_l2, use_container_width=True)
            
            st.markdown("""
            **Análise de Comportamento Dinâmico:**
            * **Curva Plana/Quase Reta:** LR muito baixa (convergência excessivamente lenta).
            * **Curva com Oscilações/Degraus Brutos:** LR muito alta (passos caóticos oscilando nas encostas do mínimo global).
            * **Curva Exponencial Suave:** Ponto ideal (*Sweet Spot*).
            """)
            
        st.divider()

        g9, g10 = st.columns(2)
        with g9:
            if "epoch_time" in hist_lr.columns:
                df_tempo_medio = hist_lr.groupby("lr_str")["epoch_time"].mean().reset_index()
                fig_l3 = px.bar(df_tempo_medio, x="lr_str", y="epoch_time", color="lr_str", text="epoch_time",
                                title="LR × Tempo Médio por Época (Segundos)", labels={"lr_str": "Learning Rate", "epoch_time": "Segundos"})
                fig_l3.update_traces(texttemplate="%{text:.3f}", textposition="auto")
                st.plotly_chart(fig_l3, use_container_width=True)
                
                st.markdown("""
                **Análise do Custo Temporal por Época:**
                A taxa de aprendizado em si não altera o número de operações matemáticas executadas. Portanto, os tempos por época devem ser estatisticamente muito próximos. Variações bruscas aqui indicam gargalos externos do sistema operacional ou flutuações de hardware durante a execução da CPU.
                """)
            else:
                st.info("A coluna 'epoch_time' não foi encontrada em historico.csv para renderizar o gráfico de tempo.")
                
        with g10:
            fig_l4 = px.bar(df_lr, x="lr_str", y="acc_por_gflop", color="lr_str", text="acc_por_gflop",
                            title="LR × Eficiência Computacional (Acc/GFLOP)", labels={"lr_str": "Learning Rate", "acc_por_gflop": "Métrica de Retorno"})
            fig_l4.update_traces(texttemplate="%{text:.4f}", textposition="auto")
            st.plotly_chart(fig_l4, use_container_width=True)
            
            st.markdown("""
            **Análise de Retorno sobre Investimento Computacional:**
            Mostra qual configuração de otimizador extraiu mais inteligência usando exatamente o mesmo orçamento computacional (já que o tamanho de camada oculta está travado em 64). Maximizar este índice indica o ajuste mais elegante possível.
            """)
            
        st.divider()

        st.subheader("📋 Diagnóstico do Otimizador")
        melhor_lr_val = df_lr.loc[df_lr["test_acc"].idxmax()]["lr"]
        st.success(f"""
        * **Configuração Recomendada:** A taxa de aprendizado **{melhor_lr_val}** entregou o melhor balanço de estabilidade matemática e acurácia de teste.
        * **Dica de Análise de Convergência:** Avalie o gráfico de linhas da Loss. Taxas muito altas tendem a ficar instáveis ou apresentar degraus bruscos, enquanto taxas excessivamente baixas geram linhas quase retas decrescentes (lentas demais para o orçamento de épocas definido).
        """)