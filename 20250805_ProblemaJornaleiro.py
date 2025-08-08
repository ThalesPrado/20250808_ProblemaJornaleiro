import streamlit as st
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Problema do Jornaleiro - Chocolates", layout="centered")

st.title("Problema do Jornaleiro - Indústria/Varejo")
st.markdown("Este app calcula a quantidade ótima de produtos que devem ser pedidos da indústria para venda no varejo.")

# Abas
aba = st.radio("Escolha uma seção:", [
    "🔧 Calculadora",
    "📘 Intuição da Modelagem",
    "📂 Etapas da Modelagem Matemática",
    "🧮 Exemplo Numérico"
])

if aba == "🔧 Calculadora":
    st.sidebar.header("Parâmetros de entrada")
    p = st.sidebar.number_input("Preço de venda SKU (R$)", value=12.0, min_value=0.0, step=0.5)
    c = st.sidebar.number_input("CMV Sku (R$)", value=5.0, min_value=0.0, step=0.5)
    s = st.sidebar.number_input("Valor SKU não vendido (R$)", value=0.0, min_value=0.0, step=0.5)
    mu = st.sidebar.number_input("Média da demanda (SKU)", value=10000, min_value=0)
    sigma = st.sidebar.number_input("Desvio padrão da demanda", value=2000, min_value=1)

    if c >= p:
        st.error("O custo de produção deve ser menor que o preço de venda para gerar lucro.")
        st.stop()
    if s > p:
        st.error("O valor residual não pode ser maior que o preço de venda.")
        st.stop()

    critical_fractile = (p - c) / (p - s)
    z_score = stats.norm.ppf(critical_fractile)
    Q_opt = mu + z_score * sigma

    st.subheader("📈 Resultado")
    st.markdown(f"**Fractil crítico (nível de serviço ótimo):** {critical_fractile:.3f}")
    st.markdown(f"**z-score correspondente:** {z_score:.3f}")
    st.markdown(f"**Quantidade ótima de barras a produzir (Q\\*):** {Q_opt:.0f}")

    # Distribuição com sombreamento duplo
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    y = stats.norm.pdf(x, mu, sigma)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, label="Distribuição da Demanda", color="black")
    ax.axvline(Q_opt, color='red', linestyle='--', label=f"Q* = {Q_opt:.0f}")
    ax.fill_between(x, y, where=(x <= Q_opt), color='green', alpha=0.3, label="Sobras (estoque)")
    ax.fill_between(x, y, where=(x > Q_opt), color='orange', alpha=0.3, label="Perda de vendas")
    ax.set_xlabel("Quantidade de Barras")
    ax.set_ylabel("Densidade de Probabilidade")
    ax.set_title("Distribuição da Demanda e Quantidade Ótima")
    ax.legend()
    st.pyplot(fig)
    
    # Lucro esperado vs Q
    st.subheader("💰 Lucro Esperado para diferentes valores de Q")
    Q_test = np.arange(mu - 3 * sigma, mu + 3 * sigma, 100)
    lucro_esperado = []
    for Q in Q_test:
        D = np.linspace(0, Q * 2, 1000)
        f = stats.norm.pdf(D, mu, sigma)
        lucro = np.where(D < Q,
                     (p - c) * D + (s - c) * (Q - D),
                     (p - c) * Q)
        lucro_esperado.append(np.trapz(lucro * f, D))

    fig3, ax3 = plt.subplots()
    ax3.plot(Q_test, lucro_esperado, label="Lucro Esperado")
    ax3.axvline(Q_opt, color='red', linestyle='--', label=f"Q* = {Q_opt:.0f} (Ótimo)")
     
    ax3.set_xlabel("Quantidade Produzida (Q)")  
    ax3.set_ylabel("Lucro Esperado (R$)")
    ax3.set_title("Lucro Esperado vs Quantidade Produzida")
    ax3.legend()
    st.pyplot(fig3)

    # Escolha de distribuição
    st.subheader("🔄 Escolher Distribuição de Demanda (Experimental)")
    dist_choice = st.selectbox("Distribuição:", ["Normal", "Lognormal","Triangular","Uniforme"])

    if dist_choice == "Lognormal":
        from scipy.stats import lognorm
        shape = sigma / mu
        scale = mu
        x = np.linspace(0, mu + 4 * sigma, 1000)
        y = lognorm.pdf(x, shape, scale=scale)
        Q_opt_logn = lognorm.ppf(critical_fractile, shape, scale=scale)
        fig4, ax4 = plt.subplots()
        ax4.plot(x, y, label="Lognormal")
        ax4.axvline(Q_opt_logn, color='red', linestyle='--', label=f"Q* ≈ {Q_opt_logn:.0f}")
        ax4.legend()
        st.pyplot(fig4)

    elif dist_choice == "Uniforme":
        from scipy.stats import uniform
        low = mu - np.sqrt(3) * sigma
        high = mu + np.sqrt(3) * sigma
        x = np.linspace(low, high, 1000)
        y = uniform.pdf(x, loc=low, scale=high - low)
        Q_opt_unif = uniform.ppf(critical_fractile, loc=low, scale=high - low)
        fig5, ax5 = plt.subplots()
        ax5.plot(x, y, label="Uniforme")
        ax5.axvline(Q_opt_unif, color='red', linestyle='--', label=f"Q* ≈ {Q_opt_unif:.0f}")
        ax5.legend()
        st.pyplot(fig5)

    elif dist_choice == "Triangular":
        from scipy.stats import triang
        c = 0.5  # pico no meio
        a = mu - np.sqrt(6) * sigma
        b = mu + np.sqrt(6) * sigma
        scale = b - a
        x = np.linspace(a, b, 1000)
        y = triang.pdf(x, c, loc=a, scale=scale)
        Q_dist = triang.ppf(critical_fractile, c, loc=a, scale=scale)
        label = "Triangular"
      
        st.markdown("""
        ### 📘 Justificativa das Distribuições:

        - **Normal**: representa demanda com influência de vários fatores independentes, assumindo simetria.
        - **Lognormal**: útil quando a demanda é sempre positiva e pode ter picos elevados (cauda longa à direita).
        - **Uniforme**: usada quando a incerteza é máxima e todos os valores dentro de um intervalo são igualmente prováveis.
        - **Triangular**: representa uma demanda com mínimo, máximo e um pico mais provável (bom quando se tem estimativas).
        """)


elif aba == "📘 Intuição da Modelagem":
    st.header("📘 Intuição da Modelagem do Problema do Jornaleiro")
    st.markdown("""
    ### 🧠 1. A Decisão Antecipada sob Incerteza
    Imagine que sua indústria vai lançar uma barra especial para o **Dia dos Namorados**. Mas você precisa decidir **hoje** quantas barras produzir — **sem saber a demanda real** do dia.

    ---

    ### ⚖️ 2. O Dilema Econômico
    | Se produzir **poucas barras**… | Se produzir **muitas barras**… |
    |-------------------------------|-------------------------------|
    | Pode **perder vendas**        | Pode sobrar chocolate e gerar **prejuízo** |

    ---

    ### 📈 3. A Demanda como Variável Aleatória
    Como a demanda não é fixa, modelamos ela como uma **variável aleatória** com distribuição normal:
    """)
    st.latex(r"D \sim \mathcal{N}(\mu, \sigma)")

    st.markdown("""
    ---

    ### ⚖️ 4. Equilíbrio Econômico com Fractil Crítico
    """)
    st.latex(r"F(Q^*) = \frac{p - c}{p - s}")

    st.markdown("""
    ---

    ### 🎯 5. Encontrando Q*
    """)
    st.latex(r"Q^* = \mu + z \cdot \sigma")

    st.markdown("""
    ---

    ### ❓ O que é valor residual (s)?
    - **s = valor residual** da barra não vendida
    - Ex: barras que sobraram e serão vendidas com **desconto**, ou **reutilizadas**
    - Se você vender por R$ 12 no dia, e liquida por R$ 2 depois, então **s = 2**

    ---

    ### ✅ 6. Por que isso importa?
    - Aplica-se a produtos sazonais como flores, roupas, ovos de Páscoa, etc.
    - Modelar a incerteza evita **perda de vendas** e **excesso de estoque**
    """)

elif aba == "📂 Etapas da Modelagem Matemática":
    st.header("📂 Etapas Formais da Modelagem - Conforme CQD/UNESP")

    st.markdown("""
    ### 1. Formulação do Problema
    Encontrar a quantidade \( Q \) que **maximiza o lucro esperado**, com base nos custos e incertezas da demanda.

    ### 2. Definição das Variáveis e Parâmetros
    - \( Q \): quantidade a ser produzida
    - \( D \): demanda aleatória
    - \( c \): custo por unidade
    - \( p \): preço de venda
    - \( s \): valor residual por unidade excedente
    """)

    st.markdown("### 3. Função Lucro (cenários)")
    st.latex(r"""
    L(Q, D) = 
    \begin{cases}
        (p - c)D + (s - c)(Q - D), & \text{se } D < Q \\
        (p - c)Q, & \text{se } D \geq Q
    \end{cases}
    """)

    st.markdown("### 4. Objetivo: Maximizar o Lucro Esperado")
    st.latex(r"""
    \mathbb{E}[L(Q)] = \int_0^Q [(p - c)d + (s - c)(Q - d)]f(d)\,dd + \int_Q^\infty (p - c)Q f(d)\,dd
    """)

    st.markdown("### 5. Solução ótima via Fractil Crítico")
    st.latex(r"F(Q^*) = \frac{p - c}{p - s}")

    st.markdown("### 6. Cálculo prático")
    st.latex(r"Q^* = \mu + z \cdot \sigma,\quad \text{onde } z = \Phi^{-1}\left(\frac{p - c}{p - s}\right)")

elif aba == "🧮 Exemplo Numérico":
    st.header("🧮 Exemplo Numérico - Indústria de Chocolate")

    st.markdown("Vamos ilustrar um exemplo completo e didático para entender o Problema do Jornaleiro com chocolates para o Dia dos Namorados.")

    st.subheader("📌 1. Parâmetros do Problema")

    st.markdown("""
    | Parâmetro | Significado | Valor |
    |-----------|-------------|-------|
    | $p$ | Preço de venda por barra | R\\$ 12,00 |
    | $c$ | Custo de produção por barra | R\\$ 5,00 |
    | $s$ | Valor residual da barra não vendida | R\\$ 2,00 |
    | $\\mu$ | Média da demanda | 10.000 unidades |
    | $\\sigma$ | Desvio padrão da demanda | 2.000 unidades |
    """, unsafe_allow_html=True)

    st.markdown("**🧾 O que é o valor residual ($s$)?**")
    st.markdown("""
    Se uma barra **não for vendida** na data comemorativa, ainda pode gerar algum valor:
    - 💸 Ser vendida com desconto (ex: liquidação)
    - ♻️ Ser reutilizada como matéria-prima
    - 🗑️ Ou ser descartada (nesse caso, $s = 0$)

    Neste exemplo, consideramos que **cada barra não vendida ainda gera R\\$ 2,00 de valor**.
    """)

    st.subheader("📉 2. Cálculo do Fractil Crítico")
    st.latex(r"F(Q^*) = \frac{p - c}{p - s} = \frac{12 - 5}{12 - 2} = \frac{7}{10} = 0{,}7")

    st.markdown("Devemos produzir uma quantidade que cubra **70% da distribuição da demanda**.")

    st.subheader("🔎 3. Encontrando o z-score")
    st.latex(r"z = \Phi^{-1}(0{,}7) \approx 0{,}524")

    st.subheader("📦 4. Cálculo da Quantidade Ótima")
    st.latex(r"Q^* = \mu + z \cdot \sigma = 10{.}000 + 0{,}524 \cdot 2{.}000 = 11.048")
    st.success("✅ Resultado: Produzir aproximadamente **11.048 barras de chocolate**.")

    st.subheader("📘 5. Análise de Lucro em Três Cenários")

    # Cenário 1 - Demanda baixa
    st.markdown("### 📉 Cenário 1 – Demanda Baixa: 9.000 unidades")

    st.markdown("""
    - **Vendidas:** 9.000 barras  
      → lucro por barra = R\\$ 12,00 - R\\$ 5,00 = **R\\$ 7,00**
    - **Sobram:** 11.048 - 9.000 = 2.048 barras  
      → prejuízo por sobra = R\\$ 5,00 - R\\$ 2,00 = **R\\$ 3,00**
    """)

    st.latex(r"\text{Lucro com vendas} = 9.000 \cdot 7 = R\$ 63.000")
    st.latex(r"\text{Prejuízo com sobras} = 2.048 \cdot (-3) = -R\$ 6.144")
    st.latex(r"\text{Lucro total} = R\$ 63.000 - R\$ 6.144 = R\$ 56.856")

    st.markdown("---")

    # Cenário 2 - Demanda média
    st.markdown("### ⚖️ Cenário 2 – Demanda Média: 10.000 unidades")

    st.markdown("""
    - **Vendidas:** 10.000 barras  
      → lucro por barra = **R\\$ 7,00**
    - **Sobram:** 11.048 - 10.000 = 1.048 barras  
      → prejuízo por sobra = **R\\$ 3,00**
    """)

    st.latex(r"\text{Lucro com vendas} = 10.000 \cdot 7 = R\$ 70.000")
    st.latex(r"\text{Prejuízo com sobras} = 1.048 \cdot (-3) = -R\$ 3.144")
    st.latex(r"\text{Lucro total} = R\$ 70.000 - R\$ 3.144 = R\$ 66.856")

    st.markdown("---")

    # Cenário 3 - Demanda alta
    st.markdown("### 📈 Cenário 3 – Demanda Alta: 12.000 unidades")

    st.markdown("""
    - **Vendidas:** 11.048 barras (tudo que foi produzido)  
      → lucro por barra = **R\\$ 7,00**
    - **Sem sobras**, mas **perdeu-se oportunidade** de vender mais 952 barras
    """)

    st.latex(r"\text{Lucro total} = 11.048 \cdot 7 = R\$ 77.336")
    st.markdown("💡 Apesar de vender tudo, a produção foi insuficiente para atender toda a demanda.")

    st.subheader("🎓 Conclusões para os Alunos")

    st.markdown("""
    - ✅ O **valor residual** reduz o prejuízo com produtos não vendidos.
    - ✅ O **nível de serviço ótimo (70%)** depende dos custos e preços, não da demanda.
    - ✅ Mesmo com incerteza, conseguimos tomar decisões quantitativas e embasadas.
    - ✅ Produzir pouco reduz perdas, mas aumenta o risco de **perder vendas**.
    - ✅ Produzir muito cobre a demanda, mas aumenta o risco de **sobras e prejuízo**.
    - 🎯 O equilíbrio vem do modelo de **fractil crítico**, aplicável a diversos setores.
    """)
