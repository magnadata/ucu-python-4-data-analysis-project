"""
Script: extraer_producto_marca_presentacion.py

Objetivo:
- Leer un CSV de precios.
- Separar la columna Producto en:
  producto_base, tipo_producto, marca, presentacion, presentacion_valor,
  presentacion_unidad y especificacion.
- Marcar registros que requieren revisión.

Uso:
    python extraer_producto_marca_presentacion.py

Ajusta las variables INPUT_CSV y OUTPUT_CSV según tu ruta.
"""

import re
import unicodedata
import numpy as np
import pandas as pd


INPUT_CSV = "df.csv"
OUTPUT_CSV = "df_productos_enriquecido.csv"
OUTPUT_MAPA = "mapa_productos_extraidos.csv"
OUTPUT_REVISION = "revision_productos.csv"


def quitar_acentos(texto):
    if pd.isna(texto):
        return np.nan

    texto = str(texto)
    texto = unicodedata.normalize("NFKD", texto)
    return "".join(ch for ch in texto if not unicodedata.combining(ch))


def normalizar_clave(texto):
    """
    Genera una versión normalizada para comparar textos.
    No se usa como texto final, sino como llave de búsqueda.
    """
    if pd.isna(texto):
        return ""

    texto = quitar_acentos(str(texto)).upper()
    texto = texto.replace("&", " Y ")
    texto = texto.replace("_", " ")

    # Separa casos como 6LTS, PAQUETE1
    texto = re.sub(r"([0-9])([A-Z])", r"\1 \2", texto)
    texto = re.sub(r"([A-Z])([0-9])", r"\1 \2", texto)

    texto = re.sub(r"[^A-Z0-9/%]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def normalizar_display(texto):
    if pd.isna(texto):
        return np.nan

    texto = str(texto).strip()
    texto = re.sub(r"\s+", " ", texto)
    return texto


PACK_WORDS = r"(ENVASE|BOLSA|PAQUETE|LATA|BOTELLA|CAJA|UNIDAD|SACHET|DISPENSER)"
UNIT_WORDS = r"(KG|KGS|KILO|KILOS|GR|GRS|G|GRAMOS|LT|LTS|L|ML|CC|CM3|MTS|US|UNIDAD|UNIDADES|DOCENA|ROLLOS)"


def extraer_presentacion_original(texto):
    """
    Extrae presentaciones como:
    - Envase 900 cc
    - Paquete 1 kg.
    - (envase 900 cc)
    - 1 kg.
    - 2,5 lts
    - 4 rollos de 30 mts.
    - 1/2 docena
    """
    if pd.isna(texto):
        return np.nan

    s = str(texto).strip()

    # 1) Presentación final entre paréntesis: "(envase 900 cc)", "(1 kg.)"
    m = re.search(r"\(([^()]*)\)\s*$", s)
    if m:
        contenido = m.group(1).strip()
        cont_key = normalizar_clave(contenido)

        if re.search(rf"\b({PACK_WORDS}|{UNIT_WORDS})\b", cont_key) or re.search(r"\b\d+", cont_key):
            return normalizar_display(contenido)

    # 2) Presentación con palabra de empaque hasta el final
    m = re.search(
        r"\b(Envase|Bolsa|Paquete|Lata|Botella|Caja|Unidad|Sachet|Dispenser)\s*[^,()]*?\d[\w\s,./]*$",
        s,
        flags=re.I
    )
    if m:
        return normalizar_display(m.group(0))

    # 3) Cantidades al final, sin palabra explícita de empaque
    patrones = [
        r"\b\d+\s*rollos?\s+de\s+\d+\s*mts?\.?\s*$",
        r"\b\d+\s*/\s*\d+\s*docena\s*$",
        r"\baprox\s*\d+\s*grs?\.?\s*$",
        r"\bX\s*\d+\s*,?\s*\d+\s*us\.?\s*$",
        r"\b\d+(?:[,.]\d+)?\s*(?:kg|kgs|grs?|g|lt|lts|l|ml|cc|cm3|mts|us|unidad|unidades)\.?\s*$",
        r"\b\d+\s*(?:us|unidad|unidades)\.?\s*$",
    ]

    for patron in patrones:
        m = re.search(patron, s, flags=re.I)
        if m:
            return normalizar_display(m.group(0))

    return np.nan


def remover_presentacion_original(texto, presentacion):
    if pd.isna(texto):
        return np.nan

    s = str(texto).strip()

    if pd.isna(presentacion) or presentacion == "":
        return normalizar_display(s)

    pres = str(presentacion).strip()

    # Caso "(presentacion)"
    pattern1 = r"\s*\(\s*" + re.escape(pres) + r"\s*\)\s*$"
    s2 = re.sub(pattern1, "", s, flags=re.I)
    if s2 != s:
        return normalizar_display(s2)

    # Caso ", presentacion"
    pattern2 = r"\s*,\s*" + re.escape(pres) + r"\s*$"
    s2 = re.sub(pattern2, "", s, flags=re.I)
    if s2 != s:
        return normalizar_display(s2)

    # Caso "presentacion" al final
    pattern3 = r"\s*" + re.escape(pres) + r"\s*$"
    s2 = re.sub(pattern3, "", s, flags=re.I)
    return normalizar_display(s2)


TIPOS_PRODUCTO = [
    "Aceite de girasol", "Aceite girasol", "Aceite de maiz", "Aceite maiz", "Aceite de soja", "Aceite soja",
    "Afeitadora descartable", "Afeitadora",
    "Agua de mesa con gas", "Agua de mesa sin gas", "Agua mineral sin gas bidon", "Agua mineral con gas", "Agua mineral sin gas",
    "Aguja vacuna", "Aguja con hueso", "Aguja sin hueso",
    "Alcohol en gel", "Alcohol rectificado", "Algodon",
    "Arroz blanco", "Arroz calidad alta", "Arroz calidad media",
    "Arvejas en conserva", "Azucar blanco", "Azucar blanca",
    "Banana", "Cafe envasado no instantaneo", "Cafe instantaneo",
    "Carne picada vacuna", "Carne picada",
    "Cerveza rubia", "Cerveza", "Champu", "Shampoo comun",
    "Chorizos mezcla sueltos", "Chorizo suelto",
    "Cinta adhesiva leuco", "Cinta adhesiva",
    "Cocoa comun", "Cocoa", "Crema facial", "Curitas",
    "Desodorante en aerosol", "Desodorante aerosol",
    "Detergente para vajilla", "Detergente comun",
    "Dulce de leche", "Dulce de membrillo",
    "Fideos secos al huevo", "Fideos secos semolados",
    "Frankfurters cortos", "Frankfurter cortos",
    "Galletitas al agua", "Galletas saladas al agua", "Gasa esteril",
    "Gaseosa tipo cola env no ret", "Gaseosa tipo cola", "Gaseosa cola",
    "Hamburguesas carne vacuna", "Hamburguesas carne vacun", "Hamburguesa carne vacuna",
    "Harina de maiz", "Harina trigo comun 0000", "Harina trigo comun 000", "Harina de trigo 0000", "Harina de trigo 000",
    "Helado familiar", "Helado comun",
    "Hipoclorito de sodio comun", "Hipoclorito de sodio", "Hipolclorito de sodio comun",
    "Huevos colorados",
    "Jabon de tocador en barra", "Jabon de tocador",
    "Jabon para ropa en polvo para maquina", "Jabon en polvo maquina", "Jabon para ropa en barra",
    "Jamon cocido no artesanal", "Jamon cocido",
    "Lechuga", "Leonesa de pollo", "Leonesa comun", "Leonesa",
    "Manteca sin sal", "Manteca", "Manzana", "Margarina con sal", "Margarina",
    "Mayonesa comun", "Mermelada de durazno",
    "Nalga vacuna", "Nalga sin hueso", "Naranja",
    "Paleta vacuna", "Paleta con hueso", "Paleta sin hueso",
    "Pan de molde lacteado", "Pan de molde", "Pan flauta",
    "Panales para adultos comun", "Panales para ninos comun", "Panales Adultos", "Panales",
    "Papa", "Papel higienico hoja simple", "Pasta dental comun", "Pasta dental",
    "Peceto vacuno", "Peceto sin hueso",
    "Perfume comun", "Perfume",
    "Pescado fresco bifes de merluza", "Pescado fresco bifes de merluz",
    "Pollo entero fresco con menudos", "Pollo entero con menudos fresco",
    "Protector solar", "Pulpa de tomate concentrada comun", "Pulpa de tomate",
    "Queso rallado",
    "Repelente de mosquitos aerosol", "Repelente de mosquitos crema", "Repelente de mosquitos spray",
    "Repelente aerosol", "Repelente crema", "Repelente spray",
    "Rueda vacuna", "Rueda con hueso",
    "Sal fina yodada fluorada", "Sal fina",
    "Talco comun", "Talco", "Tinta comun", "Tinta",
    "Toallitas y tampones comun", "Toallitas Femeninas", "Toallitas Femenias",
    "Tomate", "Vino tinto comun tetrabrick", "Vino tinto",
    "Yerba mate comun", "Yerba mate sin tipo",
    "Yogur semi descremado", "Yogur", "Zapallo",
]

TIPOS_CATALOGO = sorted(
    [(normalizar_clave(t), t) for t in TIPOS_PRODUCTO],
    key=lambda x: len(x[0]),
    reverse=True
)


MARCAS = [
    "Marca Propia", "Marcas Propias", "Precio Lider", "Rio de la Plata", "Río de la Plata", "Tienda Inglesa",
    "Tata", "Devoto", "El Dorado", "Uruguay", "Optimo", "Óptimo", "Delicia", "Condesa",
    "Gillette", "Schick", "Xtreme3", "Matutina", "Nativa", "Salus", "Vitale",
    "Sin Marca", "Bioset", "Drogueria Paysandu", "Droguería Paysandú", "Zig Zag Cisne",
    "Aruba", "Blue Patna", "Green Chef", "Pony", "Saman Blanco", "Vidarroz", "Cololo", "Cololó", "Nidemar",
    "Azucarlito", "Bella Union", "Bella Unión", "Bebe & Co", "Chana", "Chaná", "Saint", "Aguila", "Águila",
    "Patricia", "Pilsen", "Zillertal", "Dove", "Fructis", "Pantene", "Sedal", "Suave",
    "Cattivelli", "Centenario", "La Familia", "Ready Plast", "Copacabana", "Vascolet",
    "Asepxia", "Aspexia", "Revitalift", "Axe", "Rexona", "Deterjane", "Hurra Nevex", "Protergente",
    "Conaprole", "Los Nietitos", "Manjar", "Adria", "Las Acacias", "Ottonello", "Schneck", "Famosa",
    "Maestro Cubano", "El Maestro Cubano", "Coca Cola", "Coca-Cola", "Pepsi", "Pepsi-Cola", "Burgy", "Paty",
    "Gourmet", "Presto Pronta Arcor", "Puritas", "Canuelas", "Cañuelas", "Granja La Sonrisa", "Primor", "Crufi",
    "Agua Jane", "Sello Rojo", "Solucion Cristal", "Solución Cristal", "El Jefe", "Prodhin", "Super Huevo",
    "Ann Bow", "Astral Plata", "Neutrogena", "Palmolive", "Drive", "Nevex", "Skip", "Bull Dog",
    "Calcar", "Kasdorf", "Adorita", "Doriana Nueva", "Qualy", "Hellmans", "El Hogar",
    "Bimbo", "Los Sorchantes", "Pan Catalan", "Pan Catalán", "Elite", "Higienol Export", "Sin Fin",
    "Colgate", "Kolynos", "Pico Jenner", "Readysec", "Babysec", "Huggies",
    "Avicola del Oeste", "Avicolas del Oeste", "Avícola del Oeste", "Tres Arroyos",
    "Dermaglos", "Dermaglós", "Hawaiian Tropic", "Eucerin", "Soundown", "Sundown",
    "De Ley", "Artesano", "Milky", "Jupiter", "Júpiter", "Off",
    "Sek", "Torrevieja", "Urusal", "J&J", "Johnson & Johnson", "Xanapie",
    "Hornimans", "La Virginia", "President", "Excellence", "Garnier", "Issue",
    "Ladysoft", "Siempre Libre", "Faisan", "Faisán", "Santa Teresa", "Tango",
    "Baldo", "Canarias", "Del Cebador", "Parmalat", "Casapueblo", "Delice Candy"
]


def construir_catalogo_marcas(marcas):
    brand_map = {}

    for marca in marcas:
        key = normalizar_clave(marca)

        if key not in brand_map:
            brand_map[key] = marca

    # Correcciones de salida canónica
    canonical_fix = {
        normalizar_clave("Río de la Plata"): "Río de la Plata",
        normalizar_clave("Óptimo"): "Óptimo",
        normalizar_clave("Sin Marca"): "Sin marca",
        normalizar_clave("Cololó"): "Cololó",
        normalizar_clave("Bella Unión"): "Bella Unión",
        normalizar_clave("Chaná"): "Chaná",
        normalizar_clave("Águila"): "Águila",
        normalizar_clave("Coca-Cola"): "Coca-Cola",
        normalizar_clave("Pepsi-Cola"): "Pepsi-Cola",
        normalizar_clave("Cañuelas"): "Cañuelas",
        normalizar_clave("Solución Cristal"): "Solución Cristal",
        normalizar_clave("Avícola del Oeste"): "Avícola del Oeste",
        normalizar_clave("Dermaglós"): "Dermaglós",
        normalizar_clave("Júpiter"): "Júpiter",
        normalizar_clave("Johnson & Johnson"): "Johnson & Johnson",
        normalizar_clave("Faisán"): "Faisán",
    }

    brand_map.update(canonical_fix)

    return sorted(
        brand_map.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )


BRAND_CATALOGO = construir_catalogo_marcas(MARCAS)


def detectar_tipo(base):
    key = normalizar_clave(base)

    for tipo_key, tipo_disp in TIPOS_CATALOGO:
        if re.match(r"^" + re.escape(tipo_key) + r"\b", key):
            resto = key[len(tipo_key):].strip()
            return tipo_disp, resto

    return np.nan, key


def detectar_marca_desde_texto(texto_key):
    if not texto_key:
        return np.nan, texto_key

    if re.search(r"\bSIN MARCA\b", texto_key):
        rem = re.sub(r"\bSIN MARCA\b", " ", texto_key)
        return "Sin marca", re.sub(r"\s+", " ", rem).strip()

    # Preferir marca al inicio del remanente
    for key, disp in BRAND_CATALOGO:
        if re.match(r"^" + re.escape(key) + r"\b", texto_key):
            rem = re.sub(r"^" + re.escape(key) + r"\b", " ", texto_key).strip()
            return disp, re.sub(r"\s+", " ", rem).strip()

    # Si no está al inicio, buscar en cualquier posición
    matches = []

    for key, disp in BRAND_CATALOGO:
        m = re.search(r"\b" + re.escape(key) + r"\b", texto_key)

        if m:
            matches.append((m.start(), len(key), key, disp))

    if matches:
        matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
        _, _, key, disp = matches[0]
        rem = re.sub(r"\b" + re.escape(key) + r"\b", " ", texto_key)
        return disp, re.sub(r"\s+", " ", rem).strip()

    return np.nan, texto_key


def limpiar_especificacion(spec_key, tipo_key=None):
    if not spec_key:
        return np.nan

    spec = spec_key

    if tipo_key:
        tk = normalizar_clave(tipo_key)
        spec = re.sub(r"^" + re.escape(tk) + r"\b", " ", spec).strip()

    spec = re.sub(r"\b(COMUN|SIN TIPO)\b", " ", spec).strip()
    spec = re.sub(r"\s+", " ", spec).strip()

    if not spec:
        return np.nan

    return spec.title()


def extraer_valor_unidad(presentacion):
    if pd.isna(presentacion):
        return pd.Series({
            "presentacion_valor": np.nan,
            "presentacion_unidad": np.nan
        })

    k = normalizar_clave(presentacion)

    # Ejemplo: 1/2 docena
    m = re.search(r"\b(\d+)\s*/\s*(\d+)\s*(DOCENA)\b", k)
    if m:
        return pd.Series({
            "presentacion_valor": float(m.group(1)) / float(m.group(2)),
            "presentacion_unidad": "DOCENA"
        })

    # Ejemplo: 4 rollos de 30 mts.
    m = re.search(r"\b(\d+)\s*ROLLOS?\s+DE\s+\d+\s*MTS?\b", k)
    if m:
        return pd.Series({
            "presentacion_valor": float(m.group(1)),
            "presentacion_unidad": "ROLLOS"
        })

    # Cantidad + unidad
    m = re.search(
        r"\b(\d+(?:[,.]\d+)?)\s*(KG|KGS|KILO|KILOS|GRS|GR|G|LT|LTS|L|ML|CC|CM3|MTS|US|UNIDAD|UNIDADES)\b",
        k
    )

    if m:
        val = float(m.group(1).replace(",", "."))
        unit = m.group(2)

        unit_map = {
            "KGS": "KG",
            "KILO": "KG",
            "KILOS": "KG",
            "GRS": "GR",
            "G": "GR",
            "LTS": "LT",
            "L": "LT",
            "UNIDAD": "UNIDADES",
            "US": "UNIDADES",
        }

        return pd.Series({
            "presentacion_valor": val,
            "presentacion_unidad": unit_map.get(unit, unit)
        })

    return pd.Series({
        "presentacion_valor": np.nan,
        "presentacion_unidad": np.nan
    })


def enriquecer_producto(prod):
    presentacion = extraer_presentacion_original(prod)
    base = remover_presentacion_original(prod, presentacion)
    base_key = normalizar_clave(base)

    # Caso especial: en el archivo aparece como marca al inicio.
    if base_key.startswith("BEBE Y CO"):
        tipo = "Perfume"
        resto = base_key
    else:
        tipo, resto = detectar_tipo(base)

    marca, spec_key = detectar_marca_desde_texto(resto)

    # Productos usualmente sin marca explícita
    if pd.isna(marca):
        if any(base_key.startswith(p) for p in [
            "CARNE PICADA VACUNA",
            "GASA ESTERIL",
            "PESCADO FRESCO",
            "PAN FLAUTA"
        ]):
            marca = "Sin marca"
        elif any(base_key.startswith(p + " ") or base_key == p for p in [
            "BANANA", "LECHUGA", "MANZANA", "NARANJA", "PAPA", "TOMATE", "ZAPALLO"
        ]):
            marca = "Sin marca"

    # Si no se pudo detectar tipo, pero sí marca, tomar lo anterior a la marca como tipo.
    if pd.isna(tipo) and not pd.isna(marca):
        brand_key = normalizar_clave(marca)
        before = re.split(r"\b" + re.escape(brand_key) + r"\b", base_key)[0].strip()

        if before:
            tipo = before.title()

    # Caso especial Bebe & Co
    if base_key.startswith("BEBE Y CO") and not pd.isna(marca):
        brand_key = normalizar_clave(marca)
        spec_key = re.sub(r"^" + re.escape(brand_key) + r"\b", " ", base_key).strip()

    especificacion = limpiar_especificacion(spec_key, tipo)

    tipo_disp = np.nan if pd.isna(tipo) else str(tipo).strip().lower().capitalize()

    return pd.Series({
        "producto_base": base,
        "tipo_producto": tipo_disp,
        "marca": marca,
        "presentacion": presentacion,
        "especificacion": especificacion
    })


def main():
    df = pd.read_csv(INPUT_CSV)

    if "Producto" not in df.columns:
        raise ValueError("No se encontró la columna Producto en el archivo.")

    extraidas = df["Producto"].apply(enriquecer_producto)
    df_out = pd.concat([df, extraidas], axis=1)

    cantidad_unidad = df_out["presentacion"].apply(extraer_valor_unidad)
    df_out = pd.concat([df_out, cantidad_unidad], axis=1)

    df_out["producto_limpio_key"] = df_out["Producto"].apply(normalizar_clave)

    # En este archivo aparece este registro como métrica, no como producto real.
    df_out["es_producto_valido"] = df_out["Producto"].ne("Establecimientos Relevados")

    df_out["requiere_revision"] = (
        ~df_out["es_producto_valido"] |
        df_out[["tipo_producto", "marca", "presentacion"]].isna().any(axis=1)
    )

    columnas = [
        "Periodo", "Grupo", "Producto", "producto_limpio_key", "producto_base",
        "tipo_producto", "marca", "presentacion", "presentacion_valor",
        "presentacion_unidad", "especificacion", "Super", "Precio",
        "es_producto_valido", "requiere_revision"
    ]

    df_out = df_out[columnas]
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    mapa = (
        df_out
        .drop_duplicates("Producto")
        [[
            "Producto", "producto_base", "tipo_producto", "marca",
            "presentacion", "presentacion_valor", "presentacion_unidad",
            "especificacion", "es_producto_valido", "requiere_revision"
        ]]
        .sort_values(
            ["es_producto_valido", "tipo_producto", "marca"],
            ascending=[False, True, True]
        )
    )

    mapa.to_csv(OUTPUT_MAPA, index=False, encoding="utf-8-sig")

    revision = df_out[df_out["requiere_revision"]].drop_duplicates("Producto")
    revision.to_csv(OUTPUT_REVISION, index=False, encoding="utf-8-sig")

    print("Archivo enriquecido:", OUTPUT_CSV)
    print("Mapa de productos:", OUTPUT_MAPA)
    print("Archivo de revisión:", OUTPUT_REVISION)
    print()
    print("Filas:", len(df_out))
    print("Productos únicos:", df_out["Producto"].nunique())
    print("Filas que requieren revisión:", int(df_out["requiere_revision"].sum()))


if __name__ == "__main__":
    main()
