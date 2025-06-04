import streamlit as st
import pandas as pd
import re
import altair as alt
import numpy as np
import io # Importar a biblioteca io para manipulação de streams

# --- Configurações da Página Streamlit ---
st.set_page_config(
    page_title="Análise de Trading Personalizada",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Adicionar CSS customizado para tema (preto e amarelo sutil)
st.markdown(
    """
    <style>
    .reportview-container {
        background: #1a1a1a; /* Fundo preto/muito escuro */
    }
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFD700; /* Dourado/Amarelo para títulos */
    }
    .stTable, .dataframe {
        color: #FFFFFF; /* Texto branco para tabelas */
        background-color: #333333; /* Fundo cinza escuro para tabelas */
    }
    .stTable thead th {
        color: #FFD700; /* Títulos das colunas em amarelo */
    }
    /* Cores dos textos gerais */
    body {
        color: #F0F0F0;
    }
    /* Estilo para botões */
    .stButton>button {
        color: #1a1a1a;
        background-color: #FFD700;
        border-radius: 5px;
        border: 1px solid #FFD700;
    }
    .stButton>button:hover {
        background-color: #e6b800; /* Amarelo mais escuro no hover */
        border-color: #e6b800;
        color: #1a1a1a;
    }
    /* Para a imagem de fundo estática (exemplo) */
    /*
    body {
        background-image: url("https://example.com/your-background-image.jpg");
        background-size: cover;
        background-attachment: fixed;
    }
    */
    </style>
    """,
    unsafe_allow_html=True
)

# --- Lógica de Processamento de Dados (AGORA RECEBE UM ARQUIVO) ---
@st.cache_data(show_spinner=False) # Cache para carregar e processar os dados apenas uma vez
def load_and_process_data(uploaded_file):
    if uploaded_file is not None:
        try:
            # Leitura do arquivo de forma mais robusta
            df = pd.read_csv(io.BytesIO(uploaded_file.read()))

            # Validar colunas essenciais
            required_cols = ['Mercado', 'Hora de inicio', 'Data da última resolução', 'Lucro/Perda (R$)']
            if not all(col in df.columns for col in required_cols):
                st.error(f"O arquivo CSV deve conter as colunas: {', '.join(required_cols)}")
                return None, None, None, None

            # Mapeamento de meses em português para inglês
            month_mapping = {
                'jan': 'Jan', 'fev': 'Feb', 'mar': 'Mar', 'abr': 'Apr', 'mai': 'May', 'jun': 'Jun',
                'jul': 'Jul', 'ago': 'Aug', 'set': 'Sep', 'out': 'Oct', 'nov': 'Nov', 'dez': 'Dec'
            }

            def replace_months(date_str, mapping):
                date_str_lower = str(date_str).lower() # Garantir que é string e minúsculas
                for pt_month, en_month in mapping.items():
                    date_str_lower = date_str_lower.replace(pt_month, en_month)
                return date_str_lower

            # Aplica a substituição de meses, garantindo que as colunas sejam strings primeiro
            df['Hora de inicio'] = df['Hora de inicio'].astype(str).apply(lambda x: replace_months(x, month_mapping))
            df['Data da última resolução'] = df['Data da última resolução'].astype(str).apply(lambda x: replace_months(x, month_mapping))

            # Conversão para datetime
            df['Hora de inicio'] = pd.to_datetime(df['Hora de inicio'], format='%d-%b-%y %H:%M', errors='coerce')
            df['Data da última resolução'] = pd.to_datetime(df['Data da última resolução'], format='%d-%b-%y %H:%M', errors='coerce')

            # Remover linhas onde a conversão de data falhou (se houver)
            df.dropna(subset=['Hora de inicio', 'Data da última resolução'], inplace=True)
            if df.empty:
                st.error("Nenhuma data válida encontrada após a conversão. Verifique o formato das datas no arquivo.")
                return None, None, None, None
            
            # Ordenar por 'Hora de inicio' para cálculos cumulativos corretos
            df = df.sort_values(by='Hora de inicio').reset_index(drop=True)

            # --- Recálculo da Banca Acumulada Geral (para detalhes das operações) ---
            df['Banca Acumulada (R$)'] = df['Lucro/Perda (R$)'].cumsum()

            # Extração do Tipo de Mercado
            def extract_market_type(market_string):
                market_string_lower = str(market_string).lower()
                if 'resultado da partida' in market_string_lower:
                    return 'Resultado da Partida'
                elif 'mais/menos gols' in market_string_lower or 'mais/menos' in market_string_lower or 'total de gols' in market_string_lower:
                    return 'Mais/Menos Gols'
                elif 'gols no primeiro tempo' in market_string_lower:
                    return 'Gols no Primeiro Tempo'
                elif 'intervalo' in market_string_lower:
                    return 'Intervalo'
                elif 'placar correto' in market_string_lower:
                    return 'Placar Correto'
                elif 'ambas as equipes marcam' in market_string_lower:
                    return 'Ambas Equipes Marcam'
                elif 'empate anula a aposta' in market_string_lower:
                    return 'Empate Anula Aposta'
                elif 'chance dupla' in market_string_lower:
                    return 'Chance Dupla'
                elif 'handicap' in market_string_lower:
                    return 'Handicap'
                elif 'cartões' in market_string_lower:
                    return 'Cartões'
                elif 'escanteios' in market_string_lower:
                    return 'Escanteios'
                else:
                    return 'Outros'

            df['Tipo de Mercado'] = df['Mercado'].apply(extract_market_type)

            fixed_stake = 500.00
            df['Stake (R$)'] = fixed_stake
            df['Odd Calculada'] = np.where(df['Lucro/Perda (R$)'] > 0, (df['Lucro/Perda (R$)'] / fixed_stake) + 1, np.nan)
            
            # --- Nova Lógica para Banca Acumulada Diária e com Reset Mensal ---
            daily_pl = df.groupby(df['Hora de inicio'].dt.normalize())['Lucro/Perda (R$)'].sum().reset_index()
            daily_pl.columns = ['Data', 'Lucro/Perda Diário (R$)']
            
            bankroll_data_for_chart_list = []
            if not daily_pl.empty:
                daily_pl['Mes'] = daily_pl['Data'].dt.to_period('M')
                daily_pl['Banca Acumulada (R$)'] = daily_pl.groupby('Mes')['Lucro/Perda Diário (R$)'].cumsum()
                for mes, group in daily_pl.groupby('Mes'):
                    start_of_month = group['Data'].min().normalize()
                    bankroll_data_for_chart_list.append(pd.DataFrame({
                        'Data': [start_of_month],
                        'Banca Acumulada (R$)': [0.0],
                        'Mes': [mes]
                    }))
                    bankroll_data_for_chart_list.append(group[['Data', 'Banca Acumulada (R$)', 'Mes']])
            
            if bankroll_data_for_chart_list:
                bankroll_data_for_chart = pd.concat(bankroll_data_for_chart_list, ignore_index=True)
                bankroll_data_for_chart = bankroll_data_for_chart.sort_values(by='Data').reset_index(drop=True)
                bankroll_data_for_chart['Data'] = pd.to_datetime(bankroll_data_for_chart['Data'])
            else:
                bankroll_data_for_chart = pd.DataFrame(columns=['Data', 'Banca Acumulada (R$)', 'Mes'])

            # Extração e Cálculo de Lucratividade por Time
            team_profits = {}
            for index, row in df.iterrows():
                market_str = str(row['Mercado'])
                profit_loss = row['Lucro/Perda (R$)']
                match = re.search(r'Futebol \/ (.+) x (.+) :', market_str)
                if match:
                    team1 = match.group(1).strip().upper()
                    team2 = match.group(2).strip().upper()
                    team_profits[team1] = team_profits.get(team1, 0) + profit_loss
                    team_profits[team2] = team_profits.get(team2, 0) + profit_loss
            
            team_profits_df = pd.DataFrame(list(team_profits.items()), columns=['Time', 'Lucro/Prejuízo Total (R$)'])
            if not team_profits_df.empty:
                team_profits_df = team_profits_df.sort_values(by='Lucro/Prejuízo Total (R$)', ascending=False).reset_index(drop=True)
            else:
                team_profits_df = pd.DataFrame(columns=['Time', 'Lucro/Prejuízo Total (R$)'])

            return df, team_profits_df, daily_pl, bankroll_data_for_chart
        
        except pd.errors.ParserError:
            st.error("Erro de parse: Verifique se o arquivo é um CSV válido. Certifique-se de que os dados estão formatados corretamente.")
            return None, None, None, None
        except KeyError as ke:
            st.error(f"Coluna ausente no arquivo: {ke}. Por favor, verifique se o arquivo tem todas as colunas esperadas: 'Mercado', 'Hora de inicio', 'Data da última resolução', 'Lucro/Perda (R$)'.")
            return None, None, None, None
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado ao processar o arquivo: {e}. Detalhes: {type(e).__name__} - {e}")
            return None, None, None, None
    else:
        return None, None, None, None

# --- Conteúdo Principal do Aplicativo ---
st.header("Guia Rápido: Como Usar e Acessar Dados")
st.write("Bem-vindo(a)! Assista ao vídeo abaixo para entender como obter seus dados da Betfair e como navegar por esta plataforma de análise.")

st.subheader("1. Como Obter Dados da Betfair")
st.write("Aprenda o passo a passo para exportar seu histórico de Lucro/Perda da Betfair e garantir que ele esteja no formato correto para análise.")
st.video("https://youtu.be/YuwmMmhbKOw?si=cX_dLNU_rrZk2iv6") # LINK DO VÍDEO ATUALIZADO

st.write("---") 

st.header("Carregue Sua Planilha de Dados")
uploaded_file = st.file_uploader("Arraste e solte seu arquivo CSV aqui ou clique para buscar", type=["csv"])

if uploaded_file is not None:
    with st.spinner('Processando sua planilha...'):
        df, team_profits_df, daily_pl, bankroll_data_for_chart = load_and_process_data(uploaded_file)

    if df is not None and not df.empty:
        st.success("Planilha carregada e processada com sucesso!")
        
        # --- Cálculo de Métricas para Exibição ---
        total_profit_loss = df['Lucro/Perda (R$)'].sum() 
        winning_bets = df[df['Lucro/Perda (R$)'] > 0].shape[0]
        total_entries = df.shape[0]
        win_rate = (winning_bets / total_entries) * 100 if total_entries > 0 else 0
        
        # Cálculo de Lucro/Prejuízo Médio por Dia Trabalhado
        unique_days_worked = df['Hora de inicio'].dt.normalize().nunique()
        avg_profit_loss_per_day_worked = total_profit_loss / unique_days_worked if unique_days_worked > 0 else 0
        
        if not df.empty:
            if not df[df['Lucro/Perda (R$)'] > 0].empty:
                max_profit_row = df.loc[df['Lucro/Perda (R$)'].idxmax()]
            else:
                max_profit_row = {'Mercado': 'N/A', 'Hora de inicio': pd.NaT, 'Lucro/Perda (R$)': 0.0}

            if not df[df['Lucro/Perda (R$)'] < 0].empty:
                max_loss_row = df.loc[df['Lucro/Perda (R$)'].idxmin()]
            else:
                max_loss_row = {'Mercado': 'N/A', 'Hora de inicio': pd.NaT, 'Lucro/Perda (R$)': 0.0}
        else: 
            max_profit_row = {'Mercado': 'N/A', 'Hora de inicio': pd.NaT, 'Lucro/Perda (R$)': 0.0}
            max_loss_row = {'Mercado': 'N/A', 'Hora de inicio': pd.NaT, 'Lucro/Perda (R$)': 0.0}

        market_summary = df.groupby('Tipo de Mercado').agg(
            Total_Entradas=('Mercado', 'size'),
            Lucro_Prejuizo=('Lucro/Perda (R$)', 'sum'),
            Wins=('Lucro/Perda (R$)', lambda x: (x > 0).sum()),
            Losses=('Lucro/Perda (R$)', lambda x: (x < 0).sum())
        ).reset_index()
        market_summary['Taxa de Vitoria'] = (market_summary['Wins'] / market_summary['Total_Entradas'].replace(0, np.nan)) * 100
        market_summary = market_summary.sort_values(by='Lucro_Prejuizo', ascending=False)
        market_summary = market_summary[['Tipo de Mercado', 'Total_Entradas', 'Lucro_Prejuizo', 'Taxa de Vitoria']]

        st.title("💰 Análise de Trading Personalizada")
        st.write("---")

        st.header("Sumário Geral de Desempenho")
        col1, col2, col3 = st.columns(3)
        col1.metric("Lucro/Prejuízo Total", f"R$ {total_profit_loss:.2f}")
        col2.metric("Total de Entradas", total_entries)
        col3.metric("Taxa de Vitória", f"{win_rate:.1f}%")

        col4, col5, col6 = st.columns(3)
        # MÉTRICA ATUALIZADA AQUI
        col4.metric("Lucro/Prejuízo Médio por Dia Trabalhado", f"R$ {avg_profit_loss_per_day_worked:.2f}") 
        
        max_profit_date_str = max_profit_row['Hora de inicio'].strftime('%d/%m/%Y %H:%M') if pd.notna(max_profit_row['Hora de inicio']) else 'N/A'
        max_loss_date_str = max_loss_row['Hora de inicio'].strftime('%d/%m/%Y %H:%M') if pd.notna(max_loss_row['Hora de inicio']) else 'N/A'

        col5.metric("Maior Lucro", f"R$ {max_profit_row['Lucro/Perda (R$)']:.2f}")
        col5.caption(f"({max_profit_row['Mercado']} em {max_profit_date_str})")
        col6.metric("Maior Prejuízo", f"R$ {max_loss_row['Lucro/Perda (R$)']:.2f}")
        col6.caption(f"({max_loss_row['Mercado']} em {max_loss_date_str})")
        st.write("---")

        st.header("Desempenho por Tipo de Mercado")
        st.dataframe(market_summary.style.format({
            'Lucro_Prejuizo': "R$ {:.2f}",
            'Taxa de Vitoria': "{:.1f}%"
        }))
        st.write("---")

        st.header("Lucratividade por Time")
        if not team_profits_df.empty:
            st.subheader("Times Mais Lucrativos")
            st.dataframe(team_profits_df.head(5).style.format({'Lucro/Prejuízo Total (R$)': "R$ {:.2f}"}))
            st.subheader("Times Menos Lucrativos")
            st.dataframe(team_profits_df.tail(5).style.format({'Lucro/Prejuízo Total (R$)': "R$ {:.2f}"}))
        else:
            st.write("Não foram encontrados dados de times para exibir lucratividade.")
        st.write("---")

        st.header("Visualização da Banca")

        if bankroll_data_for_chart is not None and not bankroll_data_for_chart.empty:
            chart_bankroll = alt.Chart(bankroll_data_for_chart).mark_line(point=True).encode(
                x=alt.X('Data:T', title='Data', axis=alt.Axis(format='%d/%m')),
                y=alt.Y('Banca Acumulada (R$):Q', title='Banca Acumulada (R$)'),
                color=alt.Color('Mes:N', title='Mês', type='nominal'),
                tooltip=[
                    alt.Tooltip('Data:T', title='Data', format='%d/%m/%Y'),
                    alt.Tooltip('Banca Acumulada (R$):Q', title='Banca do Mês', format='.2f'),
                    alt.Tooltip('Mes:N', title='Mês')
                ]
            ).properties(
                title='Crescimento/Decrescimento da Banca (Reset Mensal)'
            ).interactive()
            st.altair_chart(chart_bankroll, use_container_width=True)
        else:
            st.write("Não há dados suficientes para exibir o gráfico de crescimento da banca.")

        if daily_pl is not None and not daily_pl.empty:
            daily_pl['Cor'] = np.where(daily_pl['Lucro/Perda Diário (R$)'] >= 0, 'Ganho', 'Perda')
            chart_daily_pl = alt.Chart(daily_pl).mark_bar(
                size=15 
            ).encode(
                x=alt.X('Data:T', title='Data', axis=alt.Axis(format='%d/%m')),
                y=alt.Y('Lucro/Perda Diário (R$):Q', title='Lucro/Perda Diário (R$)'),
                color=alt.Color('Cor:N', scale=alt.Scale(domain=['Ganho', 'Perda'], range=['green', 'red']), legend=alt.Legend(title="Resultado")),
                tooltip=[
                    alt.Tooltip('Data:T', format='%d/%m/%Y'),
                    alt.Tooltip('Lucro/Perda Diário (R$):Q', format='.2f'),
                    'Cor:N'
                ]
            ).properties(
                title='Lucro/Prejuízo Diário'
            ).interactive()
            st.altair_chart(chart_daily_pl, use_container_width=True)
        else:
            st.write("Não há dados suficientes para exibir o gráfico de lucro/prejuízo diário.")
        
        st.write("---")

        st.header("Detalhes de Todas as Operações")
        with st.expander("Clique para expandir/ocultar as operações detalhadas"):
            if df is not None and not df.empty:
                df_display = df.sort_values(by='Hora de inicio', ascending=False).reset_index(drop=True)
                df_display['Hora de inicio'] = df_display['Hora de inicio'].dt.strftime('%d/%m/%Y %H:%M')
                df_display['Data da última resolução'] = df_display['Data da última resolução'].dt.strftime('%d/%m/%Y %H:%M')

                st.dataframe(df_display[[
                    'Hora de inicio',
                    'Mercado',
                    'Tipo de Mercado',
                    'Lucro/Perda (R$)',
                    'Banca Acumulada (R$)'
                ]].style.format({
                    'Lucro/Perda (R$)': "R$ {:.2f}",
                    'Banca Acumulada (R$)': "R$ {:.2f}"
                }))
            else:
                st.write("Não há operações para detalhar.")

    elif uploaded_file is not None: 
        st.warning("Não foi possível processar os dados do arquivo enviado ou o arquivo não contém dados válidos após o processamento. Por favor, tente novamente com um arquivo CSV válido.")
else:
    st.info("Por favor, carregue seu arquivo CSV da Betfair para começar a análise.")