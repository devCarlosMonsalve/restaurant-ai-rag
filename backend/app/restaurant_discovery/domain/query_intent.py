import re
import unicodedata


def normalize_text(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .casefold()
    )
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value))


def asks_for_photos(query: str) -> bool:
    normalized_query = normalize_text(query)
    return any(
        term in normalized_query
        for term in (
            "foto",
            "fotos",
            "imagen",
            "imagenes",
            "photo",
            "photos",
            "image",
            "images",
        )
    )


def unsupported_live_data_notice(query: str) -> str | None:
    normalized_query = normalize_text(query)
    asks_availability = any(
        term in normalized_query
        for term in (
            "disponibilidad",
            "reservar",
            "reserva",
            "mesa esta noche",
            "mesa hoy",
            "reservation",
            "availability",
            "book a table",
            "table tonight",
            "available table",
        )
    )
    asks_current_price = any(
        term in normalized_query
        for term in (
            "cuanto cuesta",
            "cuanto vale",
            "precio actual",
            "precios actuales",
            "precio del menu",
            "coste del menu",
            "costo del menu",
            "how much does",
            "how much is",
            "menu price",
            "current price",
        )
    )
    if not asks_availability and not asks_current_price:
        return None

    if is_spanish_query(query):
        if asks_availability and asks_current_price:
            return (
                "No puedo verificar la disponibilidad de mesa esta noche ni "
                "los precios actuales del menú con las herramientas disponibles."
            )
        if asks_availability:
            return (
                "No puedo verificar la disponibilidad actual de mesas con las "
                "herramientas disponibles."
            )
        return (
            "No puedo verificar los precios actuales del menú con las "
            "herramientas disponibles."
        )

    if asks_availability and asks_current_price:
        return (
            "I can't verify current table availability or menu prices with "
            "the available tools."
        )
    if asks_availability:
        return "I can't verify current table availability with the available tools."
    return "I can't verify current menu prices with the available tools."


def is_spanish_query(query: str) -> bool:
    normalized_query = normalize_text(query)
    return any(
        term in normalized_query
        for term in (
            "disponibilidad",
            "reservar",
            "reserva",
            "mesa",
            "cuanto",
            "cuesta",
            "precio",
            "foto",
            "fotos",
            "imagen",
            "imagenes",
            "busca",
            "restaurante",
        )
    )
