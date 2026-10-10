from app.restaurant_discovery.domain.query_intent import (
    asks_for_photos,
    normalize_text,
    unsupported_live_data_notice,
)


def test_query_normalization_removes_accents_and_normalizes_case() -> None:
    assert normalize_text("¿Cuánto cuesta?") == "cuanto cuesta"


def test_photo_intent_matches_supported_languages() -> None:
    assert asks_for_photos("Quiero ver imágenes del restaurante")
    assert asks_for_photos("Show me photos of the restaurant")
    assert not asks_for_photos("Find an Italian restaurant")


def test_live_data_notice_preserves_spanish_and_english_wording() -> None:
    assert unsupported_live_data_notice(
        "Dime cuál tiene disponibilidad de mesa esta noche y cuánto cuesta el menú."
    ) == (
        "No puedo verificar la disponibilidad de mesa esta noche ni "
        "los precios actuales del menú con las herramientas disponibles."
    )
    assert unsupported_live_data_notice("Can I book a table tonight?") == (
        "I can't verify current table availability with the available tools."
    )
    assert unsupported_live_data_notice("Find a restaurant with Italian food") is None
