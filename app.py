
# ASISTENTE DE CANASTA FAMILIAR
#
# Proyecto Final<br>
# Elaborado Por: Luis Manuel Morales<br>
# Curso: Python for Data Analysis<br>
# Corte: Diploma Ciencia de Datos e IA Aplicada_827648_DIL.DATO_2026-1A<br>
# Instituto: UCU / URUGUAY

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# Configuración general
# ============================================================

st.set_page_config(
    page_title="EDA Interactivo - Prototipo de Calculadora de Canasta Familiar",
    page_icon="🛒",
    layout="wide"
)

sns.set_theme(style="whitegrid")


# ============================================================
# Funciones auxiliares
# ============================================================

@st.cache_data
def cargar_datos(archivo):
    """
    Carga el CSV enriquecido.
    El archivo suministrado usa separador ';'. Si falla, intenta detección automática.
    """
    try:
        df = pd.read_csv(archivo, sep=";", encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(archivo, sep=None, engine="python", encoding="utf-8-sig")

    df.columns = df.columns.str.strip()

    if "Periodo" in df.columns:
        df["Periodo"] = pd.to_datetime(df["Periodo"], errors="coerce")

    columnas_numericas = [
        "Precio",
        "presentacion_valor",
        "canasta_producto",
        "canasta_cantidad"
    ]

    for col in columnas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def formato_numero(valor, decimales=2):
    if pd.isna(valor):
        return "N/D"
    return f"{valor:,.{decimales}f}"


def resumen_descriptivo(df_filtrado):
    """
    Calcula resumen descriptivo de columnas numéricas después de aplicar filtros.
    """
    #columnas_numericas = df_filtrado.select_dtypes(include=np.number).columns.tolist()
    columnas_numericas = ['Precio']

    if not columnas_numericas:
        return pd.DataFrame()

    resumen = df_filtrado[columnas_numericas].agg([
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max"
    ]).T

    resumen["rango"] = resumen["max"] - resumen["min"]
    resumen["q1"] = df_filtrado[columnas_numericas].quantile(0.25)
    resumen["q2"] = df_filtrado[columnas_numericas].quantile(0.50)
    resumen["q3"] = df_filtrado[columnas_numericas].quantile(0.75)

    resumen = resumen[
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
            "rango",
            "q1",
            "q2",
            "q3"
        ]
    ]

    return resumen.round(2)


def obtener_df_canasta(df_base):
    """
    Filtra únicamente los registros que forman parte de la canasta.

    Regla:
    - canasta_producto = 1
    - Precio no nulo
    - canasta_cantidad no nula
    - tipo_producto no nulo
    """
    columnas_requeridas = {
        "canasta_producto",
        "canasta_cantidad",
        "Precio",
        "tipo_producto"
    }

    if not columnas_requeridas.issubset(df_base.columns):
        return pd.DataFrame()

    df_canasta = df_base[
        (df_base["canasta_producto"].fillna(0) == 1) &
        (df_base["canasta_cantidad"].notna()) &
        (df_base["Precio"].notna()) &
        (df_base["tipo_producto"].notna())
    ].copy()

    # precio canasta por producto
    df_canasta['canasta_precio'] = df_canasta['Precio'] * df_canasta['canasta_cantidad']

    return df_canasta


def calcular_canasta_agrupada_por_tipo(df_base, criterio_precio="Mediana"):
    """
    Calcula el costo de la canasta agrupando obligatoriamente por tipo_producto.

    Pasos:
    1. Filtrar registros con canasta_producto = 1.
    2. Agrupar por Periodo, Super y tipo_producto.
    3. Calcular un precio de referencia por tipo_producto.
    4. Tomar la cantidad definida para ese tipo_producto.
    5. Calcular costo_tipo_producto = canasta_cantidad * precio_referencia.
    6. Sumar los costos por Periodo y Super.

    Esta lógica evita duplicar el costo cuando un mismo tipo_producto aparece
    con varias marcas, productos o presentaciones.
    """
    df_canasta = obtener_df_canasta(df_base)

    if df_canasta.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    criterio_map = {
        "Mínimo": "min",
        "Mediana": "median",
        "Media": "mean",
        "Máximo": "max"
    }

    agg_precio = criterio_map.get(criterio_precio, "median")

    dimensiones_tipo = [
        col for col in ["Periodo", "Super", "tipo_producto"]
        if col in df_canasta.columns
    ]

    detalle_tipo = (
        df_canasta
        .groupby(dimensiones_tipo, dropna=False)
        .agg(
            precio_referencia=("Precio", agg_precio),
            precio_minimo=("Precio", "min"),
            precio_maximo=("Precio", "max"),
            precio_promedio=("Precio", "mean"),
            precio_mediano=("Precio", "median"),
            canasta_cantidad=("canasta_cantidad", "max"),
            cantidad_minima=("canasta_cantidad", "min"),
            cantidad_maxima=("canasta_cantidad", "max"),
            registros_observados=("Producto", "count"),
            productos_observados=("Producto", "nunique") if "Producto" in df_canasta.columns else ("Precio", "count"),
            marcas_observadas=("marca", "nunique") if "marca" in df_canasta.columns else ("Precio", "count"),
            canasta_costo=("canasta_precio", "sum")
        )
        .reset_index()
    )

    detalle_tipo["costo_tipo_producto"] = (
        detalle_tipo["canasta_cantidad"] * detalle_tipo["precio_referencia"]
    )

    detalle_tipo["cantidad_consistente"] = (
        detalle_tipo["cantidad_minima"] == detalle_tipo["cantidad_maxima"]
    )

    dimensiones_total = [
        col for col in ["Periodo", "Super"]
        if col in detalle_tipo.columns
    ]

    total_canasta = (
        detalle_tipo
        .groupby(dimensiones_total, dropna=False)
        .agg(
            costo_canasta=("costo_tipo_producto", "sum"),
            tipos_producto_incluidos=("tipo_producto", "nunique"),
            registros_observados=("registros_observados", "sum"),
            productos_observados=("productos_observados", "sum"),
            marcas_observadas=("marcas_observadas", "sum"),
            cantidades_consistentes=("cantidad_consistente", "all")
        )
        .reset_index()
        .sort_values("costo_canasta")
    )

    # Tabla de validación de cobertura contra el universo total de tipos de la canasta
    universo_tipo = (
        df_canasta
        .groupby("tipo_producto", dropna=False)
        .agg(
            #canasta_cantidad=("canasta_cantidad", "max"),
            #cantidad_minima=("canasta_cantidad", "min"),
            #cantidad_maxima=("canasta_cantidad", "max"),
            precio_max=("canasta_precio", "max"),
            precio_mean=("canasta_precio", "mean"),
            precio_min=("canasta_precio", "min"),
            cantidad=("canasta_cantidad", "max"),
            ref_precio_unitario=("Precio","median"),
            #registros=("Precio", "count"),
            productos_observados=("Producto", "nunique") if "Producto" in df_canasta.columns else ("Precio", "count")
        )
        .reset_index()
        .sort_values("tipo_producto")
    )

    #universo_tipo["cantidad_consistente"] = (
    #    universo_tipo["cantidad_minima"] == universo_tipo["cantidad_maxima"]
    #)

    total_tipos_esperados = universo_tipo["tipo_producto"].nunique()

    if total_tipos_esperados > 0:
        total_canasta["cobertura_tipos_pct"] = (
            total_canasta["tipos_producto_incluidos"] / total_tipos_esperados * 100
        )

    return total_canasta, detalle_tipo, universo_tipo

# ============================================================
# Cabecera cuerpo principal
# ============================================================
st.title("🛒 ASISTENTE DE CANASTA FAMILIAR")
st.html(
    "<b>Elaborado Por:</b> Ing. Luis Manuel Morales <br>" +
    "<b>Curso:</b> Diploma Ciencia de Datos e IA / 2026-1A / UCU / URUGUAY <br/>"+
    "<b>Proyecto Final:</b> Python for Data Analysis <br/>" +
    "<b>Descripción:</b> Esta aplicación permite explorar la data enriquecida de precios de los productos y canasta familiar en Uruguay con filtros dinámicos."
)


# ============================================================
# Carga de datos
# ============================================================
archivo_default = Path("df_precios_depurado_enriquecido.csv")
archivo_subido = st.file_uploader(
    "Carga el archivo CSV enriquecido",
    type=["csv"]
)

if archivo_subido is not None:
    df = cargar_datos(archivo_subido)
elif archivo_default.exists():
    df = cargar_datos(archivo_default)
else:
    st.warning(
        "Carga el archivo CSV enriquecido o coloca "
        "'df_precios_depurado_enriquecido.csv' en la carpeta (../data/processed)"
    )
    st.stop()


# ============================================================
# Sidebar de control
# ============================================================

st.sidebar.markdown(
    """
    # 🎛️ Filtros
    """
)

df_filtrado = df.copy()

# Filtro de período
if "Periodo" in df_filtrado.columns:
    periodos_disponibles = sorted(df_filtrado["Periodo"].dropna().dt.date.unique())

    if periodos_disponibles:
        periodo_min = min(periodos_disponibles)
        periodo_max = max(periodos_disponibles)

        rango_periodo = st.sidebar.date_input(
            "Rango de período",
            value=(periodo_min, periodo_max),
            min_value=periodo_min,
            max_value=periodo_max
        )

        if isinstance(rango_periodo, tuple) and len(rango_periodo) == 2:
            fecha_inicio, fecha_fin = rango_periodo

            df_filtrado = df_filtrado[
                (df_filtrado["Periodo"].dt.date >= fecha_inicio) &
                (df_filtrado["Periodo"].dt.date <= fecha_fin)
            ]

# Filtros categóricos
filtros_categoricos = [
    "Grupo",
    "Super",
    #"tipo_producto",
    "marca",
    "nombre_comercial"
]

for columna in filtros_categoricos:
    if columna in df_filtrado.columns:
        valores = sorted(df_filtrado[columna].dropna().astype(str).unique())

        seleccion = st.sidebar.multiselect(
            label=f"Filtrar por {columna}",
            options=valores,
            default=[]
        )

        if seleccion:
            df_filtrado = df_filtrado[
                df_filtrado[columna].astype(str).isin(seleccion)
            ]

# Filtro obligatorio con slider sobre columna numérica
#columnas_numericas = df_filtrado.select_dtypes(include=np.number).columns.tolist()
#print(columnas_numericas)
columnas_numericas = ['Precio']


if not columnas_numericas:
    st.error("No hay columnas numéricas disponibles para aplicar el slider.")
    st.stop()

columna_slider_default = "Precio" if "Precio" in columnas_numericas else columnas_numericas[0]

columna_slider = st.sidebar.selectbox(
    "Columna numérica para filtrar con slider",
    options=columnas_numericas,
    index=columnas_numericas.index(columna_slider_default)
)

serie_slider = df_filtrado[columna_slider].dropna()

if serie_slider.empty:
    st.warning(f"No hay valores disponibles para la columna {columna_slider}.")
    st.stop()

valor_min = float(serie_slider.min())
valor_max = float(serie_slider.max())

if valor_min == valor_max:
    st.sidebar.info(
        f"La columna {columna_slider} tiene un único valor disponible: {valor_min}."
    )
    rango_slider = (valor_min, valor_max)
else:
    rango_slider = st.sidebar.slider(
        label=f"Rango de {columna_slider}",
        min_value=valor_min,
        max_value=valor_max,
        value=(valor_min, valor_max)
    )

df_filtrado = df_filtrado[
    (df_filtrado[columna_slider] >= rango_slider[0]) &
    (df_filtrado[columna_slider] <= rango_slider[1])
]


# ============================================================
# Métricas generales
# ============================================================

st.subheader("📌 Resultado de los filtros")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Registros filtrados", f"{len(df_filtrado):,}".replace(",", "."))
col2.metric(
    "Productos únicos",
    f"{df_filtrado['Producto'].nunique():,}".replace(",", ".")
    if "Producto" in df_filtrado.columns else "N/D"
)
col3.metric(
    "Comercios únicos",
    f"{df_filtrado['Super'].nunique():,}".replace(",", ".")
    if "Super" in df_filtrado.columns else "N/D"
)
col4.metric(
    "Precio promedio",
    formato_numero(df_filtrado["Precio"].mean())
    if "Precio" in df_filtrado.columns else "N/D"
)

if df_filtrado.empty:
    st.warning("No hay datos disponibles después de aplicar los filtros.")
    st.stop()


# ============================================================
# Resumen descriptivo
# ============================================================

st.subheader("📊 Resumen descriptivo")

resumen = resumen_descriptivo(df_filtrado)

if resumen.empty:
    st.info("No hay columnas numéricas para resumir.")
else:
    st.dataframe(resumen, width='stretch')

#with st.expander("Ver muestra de datos filtrados"):
#    st.dataframe(df_filtrado.head(100), width='stretch')


# ============================================================
# Tablas adicionales
# ============================================================

st.subheader("🧾 Reportes varios")

tab1, tab2, tab3 = st.tabs([
    "Precio por comercio",
    "Precio por producto/marca",
    "Canasta familiar"
])

with tab1:


    if {"Super", "Precio"}.issubset(df_filtrado.columns):
        columnas_super = [
            col for col in ["Super", "tipo_producto", "marca", "nombre_comercial"]
            if col in df_filtrado.columns
        ]

        tabla_super = (
            df_filtrado
            .groupby(columnas_super, dropna=False)
            .agg(
                registros=("Producto", "count"),
                precio_promedio=("Precio", "mean"),
                precio_mediano=("Precio", "median"),
                precio_minimo=("Precio", "min"),
                precio_maximo=("Precio", "max"),
                desviacion=("Precio", "std")
            )
            .reset_index()
            .sort_values("Super")
        )

        st.dataframe(tabla_super.round(2), width='stretch')
    else:
        st.info("No están disponibles las columnas Super y Precio.")

with tab2:
    columnas_producto = [
        col for col in ["tipo_producto", "marca", "nombre_comercial"]
        if col in df_filtrado.columns
    ]

    if columnas_producto and "Precio" in df_filtrado.columns:
        tabla_producto = (
            df_filtrado
            .groupby(columnas_producto, dropna=False)
            .agg(
                registros=("Precio", "count"),
                precio_promedio=("Precio", "mean"),
                precio_mediano=("Precio", "median"),
                precio_minimo=("Precio", "min"),
                precio_maximo=("Precio", "max"),
                desviacion=("Precio", "std")
            )
            .reset_index()
            .sort_values("precio_mediano")
        )

        st.dataframe(tabla_producto.round(2), width='stretch')
    else:
        st.info("No hay columnas suficientes para generar la tabla por producto/marca.")

with tab3:
    st.markdown("### 🧺 Costo de canasta agrupado por tipo de producto")
    #st.markdown(
    #    """
    #    Definición de la canasta familiar:
    #
    #    1. Filtrar registros donde `canasta_producto = 1`.
    #    2. Agrupar los productos por `tipo_producto`.
    #    3. Calcular un precio de referencia para cada `tipo_producto`.
    #    4. Calcular `costo_tipo_producto = canasta_cantidad × precio_referencia`.
    #    5. Sumar los costos de todos los `tipo_producto` para obtener el costo total de la canasta.
    #    """
    #)

    columnas_requeridas = {
        "canasta_producto",
        "canasta_cantidad",
        "Precio",
        "tipo_producto"
    }

    if not columnas_requeridas.issubset(df_filtrado.columns):
        st.info(
            "No están disponibles las columnas necesarias: "
            "canasta_producto, canasta_cantidad, Precio y tipo_producto."
        )
    else:
        df_canasta_filtrada = obtener_df_canasta(df_filtrado)

        if df_canasta_filtrada.empty:
            st.info("No hay productos de canasta después de aplicar los filtros.")
        else:
            criterio_precio = st.selectbox(
                "Precio de referencia para cada tipo_producto",
                options=["Mínimo", "Mediana", "Media", "Máximo"],
                index=1,
                help=(
                    "La mediana es una buena opción de referencia porque reduce "
                    "el efecto de precios extremos."
                )
            )

            tabla_canasta, detalle_tipo, universo_tipo = calcular_canasta_agrupada_por_tipo(
                df_filtrado,
                criterio_precio=criterio_precio
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Registros canasta",
                f"{len(df_canasta_filtrada):,}".replace(",", ".")
            )
            c2.metric(
                "Tipos producto canasta",
                f"{df_canasta_filtrada['tipo_producto'].nunique():,}".replace(",", ".")
            )
            c3.metric(
                "Productos observados",
                f"{df_canasta_filtrada['Producto'].nunique():,}".replace(",", ".")
                if "Producto" in df_canasta_filtrada.columns else "N/D"
            )
            c4.metric(
                "Criterio precio",
                criterio_precio
            )

            #st.markdown("#### Costo total de canasta por período y comercio")

            #st.dataframe(
            #    tabla_canasta.round(2),
            #    width='stretch'
            #)

            if {"Super", "costo_canasta"}.issubset(tabla_canasta.columns):
                ranking_super = (
                    tabla_canasta
                    .groupby("Super", dropna=False)
                    .agg(
                        canasta_minimo=("costo_canasta", "min"),
                        canasta_promedio=("costo_canasta", "mean"),
                        canasta_mediano=("costo_canasta", "median"),
                        canasta_maximo=("costo_canasta", "max"),
                        periodos_observados=("Periodo", "count"),
                        cobertura_tipos_promedio=("cobertura_tipos_pct", "mean")
                        #if "cobertura_tipos_pct" in tabla_canasta.columns
                        #else ("costo_canasta", "count")
                    )
                    .reset_index()
                    .sort_values("canasta_mediano")
                )

                st.markdown("#### Ranking de comercios por costo mediano de canasta")
                #st.dataframe(ranking_super.round(2), width='stretch')

                fig_canasta, ax_canasta = plt.subplots(figsize=(10, 5))
                sns.barplot(
                    data=ranking_super,
                    x="canasta_mediano",
                    y="Super",
                    ax=ax_canasta
                )
                ax_canasta.set_title(
                    f"Costo mediano de canasta por comercio - criterio {criterio_precio}"
                )
                ax_canasta.set_xlabel("Costo mediano de canasta")
                ax_canasta.set_ylabel("Comercio")
                st.pyplot(fig_canasta)

            if {"Periodo", "costo_canasta"}.issubset(tabla_canasta.columns):
                st.markdown("#### Evolución del costo de canasta")

                fig_linea, ax_linea = plt.subplots(figsize=(11, 5))

                if "Super" in tabla_canasta.columns:
                    sns.lineplot(
                        data=tabla_canasta,
                        x="Periodo",
                        y="costo_canasta",
                        hue="Super",
                        marker="o",
                        ax=ax_linea
                    )
                else:
                    sns.lineplot(
                        data=tabla_canasta,
                        x="Periodo",
                        y="costo_canasta",
                        marker="o",
                        ax=ax_linea
                    )

                ax_linea.set_title("Evolución del costo total de la canasta")
                ax_linea.set_xlabel("Periodo")
                ax_linea.set_ylabel("Costo canasta")
                plt.xticks(rotation=45)
                st.pyplot(fig_linea)

            #with st.expander("Ver detalle del cálculo por tipo_producto"):
            #    columnas_detalle = [
            #        col for col in [
            #            "Periodo",
            #            "Super",
            #            "tipo_producto",
            #            "precio_referencia",
            #            "precio_minimo",
            #            "precio_maximo",
            #            #"precio_promedio",
            #            #"precio_mediano",
            #            "canasta_cantidad",
            #            "costo_tipo_producto",
                        #"registros_observados",
                        #"productos_observados",
                        #"marcas_observadas",
                        #"cantidad_consistente"
            #        ]
            #        if col in detalle_tipo.columns
            #    ]

            #    st.dataframe(
            #        detalle_tipo[columnas_detalle].round(2),
            #        width='stretch'
            #    )

            st.markdown("#### Referencia canasta familiar por tipo de producto")
            st.dataframe(
                universo_tipo.round(2),
                width='stretch'
            )
            #with st.expander("Referencia canasta"):
                #st.dataframe(
                #    universo_tipo.round(2),
                #    width='stretch'
                #)

                #inconsistencias = universo_tipo[
                #    universo_tipo["cantidad_consistente"] == False
                #]

                #if not inconsistencias.empty:
                #    st.warning(
                #        "Hay tipos de producto con más de una cantidad definida en canasta_cantidad. "
                #        "Revisa estos casos porque pueden afectar el costo total."
                #    )

                #    st.dataframe(
                #        inconsistencias.round(2),
                #        width='stretch'
                #    )
