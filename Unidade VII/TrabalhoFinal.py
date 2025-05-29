import datetime
import tkinter as tk
import webbrowser
from tkinter import ttk, messagebox

import customtkinter as ctk
import folium
import matplotlib.pyplot as plt
import pandas as pd
from folium.plugins import MarkerCluster
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options

# Configurar tema da interface (claro por padrão)
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Dados iniciais
incidentes = []


# Função para extrair nível do rio do site AlertaBlu com Selenium
def obter_nivel_rio_web():
    try:
        # Exibir mensagem de carregando
        label_status_rio.configure(text="Nível do Rio: Carregando...", text_color="black")
        root.update()  # Atualizar interface para mostrar "Carregando"

        # Configurar Selenium com Microsoft Edge
        edge_options = Options()
        edge_options.add_argument("--headless")  # Executar sem abrir janela (opcional)
        edge_options.add_argument("--disable-gpu")
        edge_options.add_argument("--no-sandbox")
        driver = webdriver.Edge(options=edge_options)

        url = "https://alertablu.blumenau.sc.gov.br/d/nivel-do-rio"
        driver.get(url)

        # Localizar tabela usando XPath
        table = driver.find_element(By.XPATH, "/html/body/main/section/div/div/div/div/div/div/div/div/table")
        if not table:
            raise ValueError("Tabela não encontrada no site")

        # Extrair linhas da tabela
        rows = table.find_elements(By.TAG_NAME, "tr")
        if len(rows) < 2:
            raise ValueError("Tabela vazia ou sem dados")

        # Extrair células da primeira linha de dados (ignorar cabeçalho)
        cells = rows[1].find_elements(By.TAG_NAME, "td")
        if len(cells) < 2:
            raise ValueError("Dados do nível do rio não encontrados na tabela")

        nivel_rio = float(cells[1].text.strip().replace(",", "."))
        driver.quit()

        # Determinar categoria e cor
        if nivel_rio <= 3.0:
            status = f"Nível do Rio: {nivel_rio:.2f} m (Normalidade)"
            cor = "green"
        elif 3.0 < nivel_rio <= 4.0:
            status = f"Nível do Rio: {nivel_rio:.2f} m (Observação)"
            cor = "blue"
        elif 4.0 < nivel_rio <= 6.0:
            status = f"Nível do Rio: {nivel_rio:.2f} m (Atenção)"
            cor = "yellow"
        elif 6.0 < nivel_rio <= 8.0:
            status = f"Nível do Rio: {nivel_rio:.2f} m (Alerta)"
            cor = "orange"
        else:
            status = f"Nível do Rio: {nivel_rio:.2f} m (Alerta Máximo)"
            cor = "red"

        label_status_rio.configure(text=status, text_color=cor)
        messagebox.showinfo("Sucesso", f"Nível do rio atualizado: {nivel_rio:.2f} m")
        return nivel_rio
    except Exception as e:
        label_status_rio.configure(text="Nível do Rio: Erro ao carregar", text_color="red")
        messagebox.showerror("Erro", f"Falha ao obter nível do rio: {str(e)}")
        if 'driver' in locals():
            driver.quit()
        return 0.0


# Função para alternar tema
def alternar_tema():
    try:
        current_mode = ctk.get_appearance_mode()
        new_mode = "dark" if current_mode.lower() == "light" else "light"
        ctk.set_appearance_mode(new_mode)
        botao_tema.configure(text=f"Tema: {'Escuro' if new_mode == 'light' else 'Claro'}")
        messagebox.showinfo("Sucesso", f"Tema alterado para {new_mode.capitalize()}")
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao alternar tema: {str(e)}")


# Função para exibir ajuda
def exibir_ajuda():
    ajuda_texto = (
        "Sistema de Gerenciamento de Incidentes - Defesa Civil Blumenau\n\n"
        "Como usar:\n"
        "1. Cadastro: Preencha os campos (data, tipo, local, latitude, longitude, severidade) e clique em 'Cadastrar Incidente'. O nível do rio é obtido automaticamente do site AlertaBlu.\n"
        "2. Mapa: Clique em 'Gerar Mapa Interativo' para visualizar incidentes em um mapa no navegador.\n"
        "3. Gráficos: Clique em 'Gerar Gráficos' para exibir estatísticas de incidentes.\n"
        "4. Relatório: Clique em 'Gerar Relatório PDF' para criar um relatório com resumo, gráficos e incidentes.\n"
        "5. Nível do Rio: Clique em 'Atualizar Nível do Rio' para obter o nível atual do Rio Itajaí-Açu.\n"
        "6. Tema: Alterne entre tema claro e escuro com o botão 'Tema'.\n\n"
        "Para suporte, contate: suporte@defesacivil.blumenau.sc.gov.br"
    )
    messagebox.showinfo("Ajuda", ajuda_texto)


# Função para exibir LGPD
def exibir_lgpd():
    lgpd_texto = (
        "Conformidade com a LGPD\n\n"
        "Este sistema está em conformidade com a Lei Geral de Proteção de Dados (Lei nº 13.709/2018).\n"
        "- Dados coletados: Apenas informações de incidentes (data, tipo, local, coordenadas, severidade, nível do rio) são armazenadas localmente em um arquivo CSV.\n"
        "- Finalidade: Os dados são usados exclusivamente para monitoramento e gestão de incidentes pela Defesa Civil de Blumenau.\n"
        "- Segurança: Os dados são salvos localmente e não são compartilhados com terceiros.\n"
        "- Direitos do titular: Para acesso, correção ou exclusão de dados, contate o encarregado de dados da Defesa Civil: dpo@blumenau.sc.gov.br.\n"
        "- Consentimento: O uso deste sistema implica consentimento para o armazenamento temporário dos dados informados."
    )
    messagebox.showinfo("LGPD", lgpd_texto)


# Função para salvar dados em CSV
def salvar_dados():
    try:
        df = pd.DataFrame(incidentes,
                          columns=["Data", "Tipo", "Local", "Latitude", "Longitude", "Severidade", "Nível Rio (m)"])
        df.to_csv("incidentes_blumenau.csv", index=False)
        return True
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao salvar dados: {str(e)}")
        return False


# Função para carregar dados de CSV
def carregar_dados():
    global incidentes
    try:
        df = pd.read_csv("incidentes_blumenau.csv")
        incidentes = df.values.tolist()
        messagebox.showinfo("Sucesso", "Dados carregados com sucesso!")
    except FileNotFoundError:
        incidentes = []
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao carregar dados: {str(e)}")


# Função para cadastrar incidente
def cadastrar_incidente():
    try:
        data = entry_data.get()
        tipo = combo_tipo.get()
        local = entry_local.get()
        lat = float(entry_lat.get())
        lon = float(entry_lon.get())
        severidade = combo_severidade.get()
        nivel_rio = obter_nivel_rio_web()
        if data and tipo and local and lat and lon and severidade:
            incidentes.append([data, tipo, local, lat, lon, severidade, nivel_rio])
            if salvar_dados():
                messagebox.showinfo("Sucesso", f"Incidente cadastrado! Nível do rio: {nivel_rio:.2f} m")
                atualizar_lista_incidentes()
                limpar_campos()
        else:
            messagebox.showwarning("Erro", "Preencha todos os campos obrigatórios!")
    except ValueError:
        messagebox.showerror("Erro", "Latitude e Longitude devem ser números válidos!")
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao cadastrar incidente: {str(e)}")


# Função para limpar campos do formulário
def limpar_campos():
    try:
        entry_data.delete(0, tk.END)
        entry_data.insert(0, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        combo_tipo.set("")
        entry_local.delete(0, tk.END)
        entry_lat.delete(0, tk.END)
        entry_lon.delete(0, tk.END)
        combo_severidade.set("")
        messagebox.showinfo("Sucesso", "Campos limpos com sucesso!")
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao limpar campos: {str(e)}")


# Função para atualizar lista de incidentes
def atualizar_lista_incidentes():
    try:
        lista_incidentes.delete(*lista_incidentes.get_children())
        for i, incidente in enumerate(incidentes, 1):
            lista_incidentes.insert("", "end", values=(i, *incidente))
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao atualizar lista: {str(e)}")


# Função para gerar mapa interativo
def gerar_mapa():
    try:
        mapa = folium.Map(location=[-26.9194, -49.0661], zoom_start=12)  # Centro de Blumenau
        marker_cluster = MarkerCluster().add_to(mapa)

        for incidente in incidentes:
            lat, lon, severidade, nivel_rio = incidente[3], incidente[4], incidente[5], incidente[6]
            cor = {"Baixa": "green", "Média": "orange", "Alta": "red", "Crítica": "purple"}.get(severidade, "blue")
            folium.Marker(
                location=[lat, lon],
                popup=f"{incidente[1]} - {incidente[2]}<br>Severidade: {severidade}<br>Data: {incidente[0]}<br>Nível Rio: {nivel_rio:.2f} m",
                icon=folium.Icon(color=cor)
            ).add_to(marker_cluster)

        mapa.save("mapa_incidentes_blumenau.html")
        webbrowser.open("mapa_incidentes_blumenau.html")
        messagebox.showinfo("Sucesso", "Mapa gerado e aberto no navegador!")
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao gerar mapa: {str(e)}")


# Função para gerar gráficos
def gerar_graficos():
    try:
        if not incidentes:
            messagebox.showwarning("Erro", "Nenhum incidente cadastrado!")
            return

        df = pd.DataFrame(incidentes,
                          columns=["Data", "Tipo", "Local", "Latitude", "Longitude", "Severidade", "Nível Rio (m)"])

        # Limpar frame de gráficos
        for widget in frame_graficos.winfo_children():
            widget.destroy()

        # Gráfico de barras por tipo de incidente e severidade
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Contagem por tipo
        df["Tipo"].value_counts().plot(kind="bar", ax=ax1, color="#1f77b4")
        ax1.set_title("Incidentes por Tipo", fontsize=12)
        ax1.set_xlabel("Tipo", fontsize=10)
        ax1.set_ylabel("Quantidade", fontsize=10)

        # Contagem por severidade
        df["Severidade"].value_counts().plot(kind="bar", ax=ax2, color="#ff7f0e")
        ax2.set_title("Incidentes por Severidade", fontsize=12)
        ax2.set_xlabel("Severidade", fontsize=10)
        ax2.set_ylabel("Quantidade", fontsize=10)

        plt.tight_layout()

        # Incorporar gráfico na interface
        canvas = FigureCanvasTkAgg(fig, master=frame_graficos)
        canvas.draw()
        canvas.get_tk_widget().pack()
        messagebox.showinfo("Sucesso", "Gráficos gerados com sucesso!")
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao gerar gráficos: {str(e)}")


# Função para gerar relatório em PDF
def gerar_relatorio_pdf():
    try:
        if not incidentes:
            messagebox.showwarning("Erro", "Nenhum incidente cadastrado!")
            return

        df = pd.DataFrame(incidentes,
                          columns=["Data", "Tipo", "Local", "Latitude", "Longitude", "Severidade", "Nível Rio (m)"])
        pdf_file = "relatorio_defesa_civil_blumenau.pdf"
        doc = SimpleDocTemplate(pdf_file, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Título
        story.append(Paragraph("Relatório de Incidentes - Defesa Civil Blumenau", styles["Title"]))
        story.append(Spacer(1, 12))

        # Resumo
        nivel_rio_atual = obter_nivel_rio_web()
        resumo = f"Total de Incidentes: {len(df)}<br>" \
                 f"Tipos de Incidentes: {', '.join(df['Tipo'].unique())}<br>" \
                 f"Severidade Crítica: {len(df[df['Severidade'] == 'Crítica'])} incidentes<br>" \
                 f"Nível Atual do Rio Itajaí-Açu: {nivel_rio_atual:.2f} metros"
        story.append(Paragraph(resumo, styles["Normal"]))
        story.append(Spacer(1, 12))

        # Gráficos
        gerar_graficos()
        plt.savefig("graficos_blumenau.png")
        story.append(Image("graficos_blumenau.png", width=400, height=200))
        story.append(Spacer(1, 12))

        # Lista de incidentes
        for _, row in df.iterrows():
            texto = f"Data: {row['Data']}, Tipo: {row['Tipo']}, Local: {row['Local']}, " \
                    f"Lat: {row['Latitude']}, Lon: {row['Longitude']}, Severidade: {row['Severidade']}, " \
                    f"Nível Rio: {row['Nível Rio (m)']:.2f} m"
            story.append(Paragraph(texto, styles["Normal"]))
            story.append(Spacer(1, 6))

        # Rodapé do PDF
        story.append(Spacer(1, 12))
        story.append(Paragraph(
            "Sistema desenvolvido para a Defesa Civil de Blumenau. Versão 1.0, 2025. Contato: suporte@defesacivil.blumenau.sc.gov.br",
            styles["Normal"]))

        doc.build(story)
        messagebox.showinfo("Sucesso", f"Relatório gerado! Arquivo salvo como '{pdf_file}'")
    except Exception as e:
        messagebox.showerror("Erro", f"Falha ao gerar relatório: {str(e)}")


# Interface gráfica
root = ctk.CTk()
root.title("Sistema de Gerenciamento de Incidentes - Defesa Civil Blumenau")
root.geometry("1000x800")

# Menu
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)
ajuda_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Ajuda", menu=ajuda_menu)
ajuda_menu.add_command(label="Como Usar", command=exibir_ajuda)
ajuda_menu.add_command(label="LGPD", command=exibir_lgpd)

# Frame para título e resumo
frame_titulo = ctk.CTkFrame(root, corner_radius=10)
frame_titulo.pack(pady=10, padx=10, fill="x")
ctk.CTkLabel(frame_titulo, text="Sistema de Gerenciamento de Incidentes - Defesa Civil Blumenau",
             font=("Arial", 20, "bold")).pack(pady=5)
resumo_texto = (
    "Este sistema auxilia a Defesa Civil de Blumenau no monitoramento de incidentes (enchentes, deslizamentos, etc.) "
    "com integração ao AlertaBlu para obter o nível do Rio Itajaí-Açu em tempo real. "
    "Funcionalidades incluem cadastro de incidentes, mapas interativos, gráficos e relatórios em PDF. "
    "Os dados são salvos localmente em CSV, em conformidade com a LGPD."
)
ctk.CTkLabel(frame_titulo, text=resumo_texto, wraplength=950, font=("Arial", 12)).pack(pady=5)

# Frame para status do rio e tema
frame_status = ctk.CTkFrame(root, corner_radius=10)
frame_status.pack(pady=10, padx=10, fill="x")
label_status_rio = ctk.CTkLabel(frame_status, text="Nível do Rio: Carregando...", font=("Arial", 16, "bold"))
label_status_rio.pack(pady=10)
ctk.CTkButton(frame_status, text="Atualizar Nível do Rio", command=obter_nivel_rio_web).pack(pady=5)
botao_tema = ctk.CTkButton(frame_status, text="Tema: Escuro", command=alternar_tema)
botao_tema.pack(pady=5)

# Frame para formulário
frame_form = ctk.CTkFrame(root, corner_radius=10)
frame_form.pack(pady=10, padx=10, fill="x")

# Campos do formulário
ctk.CTkLabel(frame_form, text="Data (YYYY-MM-DD HH:MM):").grid(row=0, column=0, padx=10, pady=5, sticky="e")
entry_data = ctk.CTkEntry(frame_form, width=200)
entry_data.grid(row=0, column=1, padx=10, pady=5)
entry_data.insert(0, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

ctk.CTkLabel(frame_form, text="Tipo de Incidente:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
combo_tipo = ctk.CTkComboBox(frame_form,
                             values=["Enchente", "Deslizamento", "Alagamento", "Queda de Árvore", "Destelhamento"],
                             width=200)
combo_tipo.grid(row=1, column=1, padx=10, pady=5)

ctk.CTkLabel(frame_form, text="Local (Rua/Bairro):").grid(row=2, column=0, padx=10, pady=5, sticky="e")
entry_local = ctk.CTkEntry(frame_form, width=200)
entry_local.grid(row=2, column=1, padx=10, pady=5)

ctk.CTkLabel(frame_form, text="Latitude:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
entry_lat = ctk.CTkEntry(frame_form, width=200)
entry_lat.grid(row=3, column=1, padx=10, pady=5)

ctk.CTkLabel(frame_form, text="Longitude:").grid(row=4, column=0, padx=10, pady=5, sticky="e")
entry_lon = ctk.CTkEntry(frame_form, width=200)
entry_lon.grid(row=4, column=1, padx=10, pady=5)

ctk.CTkLabel(frame_form, text="Severidade:").grid(row=5, column=0, padx=10, pady=5, sticky="e")
combo_severidade = ctk.CTkComboBox(frame_form, values=["Baixa", "Média", "Alta", "Crítica"], width=200)
combo_severidade.grid(row=5, column=1, padx=10, pady=5)

# Botões
frame_botoes = ctk.CTkFrame(root, corner_radius=10)
frame_botoes.pack(pady=10, padx=10, fill="x")
ctk.CTkButton(frame_botoes, text="Cadastrar Incidente", command=cadastrar_incidente, fg_color="#1f77b4").grid(row=0,
                                                                                                              column=0,
                                                                                                              padx=10,
                                                                                                              pady=5)
ctk.CTkButton(frame_botoes, text="Gerar Mapa Interativo", command=gerar_mapa, fg_color="#2ca02c").grid(row=0, column=1,
                                                                                                       padx=10, pady=5)
ctk.CTkButton(frame_botoes, text="Gerar Gráficos", command=gerar_graficos, fg_color="#ff7f0e").grid(row=0, column=2,
                                                                                                    padx=10, pady=5)
ctk.CTkButton(frame_botoes, text="Gerar Relatório PDF", command=gerar_relatorio_pdf, fg_color="#d62728").grid(row=0,
                                                                                                              column=3,
                                                                                                              padx=10,
                                                                                                              pady=5)

# Lista de incidentes
frame_lista = ctk.CTkFrame(root, corner_radius=10)
frame_lista.pack(pady=10, padx=10, fill="both", expand=True)
colunas = ("ID", "Data", "Tipo", "Local", "Latitude", "Longitude", "Severidade", "Nível Rio (m)")
lista_incidentes = ttk.Treeview(frame_lista, columns=colunas, show="headings", height=8)
for col in colunas:
    lista_incidentes.heading(col, text=col)
    lista_incidentes.column(col, width=100)
lista_incidentes.pack(padx=10, pady=10, fill="both", expand=True)

# Frame para gráficos
frame_graficos = ctk.CTkFrame(root, corner_radius=10)
frame_graficos.pack(pady=10, padx=10, fill="both", expand=True)

# Rodapé
frame_rodape = ctk.CTkFrame(root, corner_radius=10)
frame_rodape.pack(pady=10, padx=10, fill="x")
ctk.CTkLabel(frame_rodape,
             text="Sistema desenvolvido para a Defesa Civil de Blumenau | Versão 1.0, 2025 | Contato: suporte@defesacivil.blumenau.sc.gov.br",
             font=("Arial", 10)).pack(pady=5)

# Carregar dados iniciais e atualizar status do rio
carregar_dados()
atualizar_lista_incidentes()
obter_nivel_rio_web()

# Iniciar aplicação
root.mainloop()
