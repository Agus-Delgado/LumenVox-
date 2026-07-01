"""Generate a synthetic customer feedback dataset for LumenVox."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

CHANNELS = [
    "support_ticket",
    "review",
    "nps_survey",
    "chat",
    "email",
    "contact_form",
]

CUSTOMER_SEGMENTS = [
    "freelancer",
    "small_business",
    "startup",
    "enterprise",
    "agency",
]

PLAN_TYPES = ["free", "basic", "pro", "enterprise"]

PRODUCT_AREAS = [
    "billing",
    "performance",
    "usability",
    "support",
    "pricing",
    "features",
    "onboarding",
    "exports",
    "integrations",
    "account_access",
]

SENTIMENT_LABELS = ["positive", "neutral", "negative"]
PRIORITIES = ["low", "medium", "high", "critical"]

HIGH_PRIORITY_AREAS = {"billing", "account_access", "performance", "integrations"}
CRITICAL_PRIORITY_AREAS = HIGH_PRIORITY_AREAS

CHANNEL_SENTIMENT_WEIGHTS = {
    "review": {"positive": 0.60, "neutral": 0.25, "negative": 0.15},
    "nps_survey": {"positive": 0.60, "neutral": 0.25, "negative": 0.15},
    "support_ticket": {"positive": 0.15, "neutral": 0.25, "negative": 0.60},
    "chat": {"positive": 0.15, "neutral": 0.25, "negative": 0.60},
    "email": {"positive": 0.15, "neutral": 0.25, "negative": 0.60},
    "contact_form": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
}

PRIORITY_AREA_WEIGHTS = {
    "default": {"low": 0.35, "medium": 0.40, "high": 0.20, "critical": 0.05},
    "high_risk": {"low": 0.10, "medium": 0.25, "high": 0.40, "critical": 0.25},
}

RESPONSE_TIME_RANGES = {
    "low": (2, 24),
    "medium": (8, 48),
    "high": (24, 96),
    "critical": (48, 168),
}

SEGMENT_SUFFIXES_EN = {
    "freelancer": "As a freelancer, this matters a lot to my workflow.",
    "small_business": "Our small team depends on this working reliably.",
    "startup": "We're scaling fast and need this resolved quickly.",
    "enterprise": "This affects multiple departments across our organization.",
    "agency": "We manage several client accounts and this is blocking delivery.",
}

SEGMENT_SUFFIXES_ES = {
    "freelancer": "Como freelancer, esto afecta mucho mi flujo de trabajo.",
    "small_business": "Nuestro equipo pequeño depende de que esto funcione bien.",
    "startup": "Estamos creciendo rápido y necesitamos una solución pronto.",
    "enterprise": "Esto impacta a varios departamentos de nuestra organización.",
    "agency": "Gestionamos varias cuentas de clientes y esto nos está frenando.",
}

PLAN_SUFFIXES_EN = {
    "free": "We're on the free plan but still expect basic reliability.",
    "basic": "Given our basic plan, we hoped for smoother support.",
    "pro": "We upgraded to Pro specifically for better service.",
    "enterprise": "As enterprise customers, we expect priority handling.",
}

PLAN_SUFFIXES_ES = {
    "free": "Estamos en el plan gratuito pero esperamos un mínimo de estabilidad.",
    "basic": "Con el plan básico, esperábamos un soporte más ágil.",
    "pro": "Pasamos al plan Pro justamente por un mejor servicio.",
    "enterprise": "Como clientes enterprise, esperamos atención prioritaria.",
}

MESSAGE_TEMPLATES: dict[str, dict[str, dict[str, list[str]]]] = {
    "billing": {
        "positive": {
            "en": [
                "Invoices are clear and payments process without issues.",
                "Billing is transparent and easy to understand each month.",
                "We appreciate the detailed breakdown on every invoice.",
                "Payment reminders are helpful and never feel intrusive.",
            ],
            "es": [
                "Las facturas son claras y los pagos se procesan sin problemas.",
                "La facturación es transparente y fácil de entender cada mes.",
                "Valoramos el desglose detallado en cada factura.",
                "Los recordatorios de pago son útiles y no resultan molestos.",
            ],
        },
        "neutral": {
            "en": [
                "Billing works fine, though the invoice layout could be clearer.",
                "Payments go through, but I'd like more payment method options.",
                "Invoices are acceptable; nothing stands out as exceptional.",
                "The billing section is functional but not particularly intuitive.",
            ],
            "es": [
                "La facturación funciona bien, aunque el diseño de facturas podría mejorar.",
                "Los pagos se procesan, pero me gustaría más opciones de pago.",
                "Las facturas son aceptables; nada destaca especialmente.",
                "La sección de facturación es funcional pero no muy intuitiva.",
            ],
        },
        "negative": {
            "en": [
                "I was charged twice this month and support hasn't responded.",
                "My subscription renewed without warning and I can't get a refund.",
                "The invoice total doesn't match what I agreed to at checkout.",
                "We've been overcharged on our enterprise plan for three months.",
            ],
            "es": [
                "Me cobraron dos veces este mes y el soporte no ha respondido.",
                "Mi suscripción se renovó sin aviso y no puedo obtener un reembolso.",
                "El total de la factura no coincide con lo acordado al pagar.",
                "Nos han cobrado de más en el plan enterprise durante tres meses.",
            ],
        },
    },
    "performance": {
        "positive": {
            "en": [
                "The platform is fast and handles our workload without lag.",
                "Page load times are consistently under two seconds for our team.",
                "Performance has been rock solid even during peak usage.",
                "Reports generate quickly even with large datasets.",
            ],
            "es": [
                "La plataforma es rápida y maneja nuestra carga sin demoras.",
                "Los tiempos de carga son consistentemente bajos para nuestro equipo.",
                "El rendimiento ha sido muy estable incluso en horas pico.",
                "Los reportes se generan rápido incluso con grandes volúmenes de datos.",
            ],
        },
        "neutral": {
            "en": [
                "Performance is acceptable most days, with occasional slowdowns.",
                "The app runs fine, though dashboards take a few extra seconds.",
                "Speed is average compared to similar tools we've used.",
                "We notice minor delays during busy hours but nothing critical.",
            ],
            "es": [
                "El rendimiento es aceptable la mayoría de los días, con lentitudes ocasionales.",
                "La app funciona bien, aunque los dashboards tardan un poco más.",
                "La velocidad es promedio comparada con herramientas similares.",
                "Notamos demoras menores en horas pico pero nada crítico.",
            ],
        },
        "negative": {
            "en": [
                "The app crashes constantly when loading large reports.",
                "We've experienced severe timeouts during business-critical operations.",
                "Performance degraded sharply after the last update.",
                "Our team loses hours every week waiting for pages to load.",
            ],
            "es": [
                "La aplicación se cae constantemente al cargar reportes grandes.",
                "Hemos tenido timeouts severos durante operaciones críticas.",
                "El rendimiento empeoró notablemente tras la última actualización.",
                "Perdemos horas cada semana esperando que carguen las páginas.",
            ],
        },
    },
    "usability": {
        "positive": {
            "en": [
                "The interface is very intuitive; I configured everything in minutes.",
                "Navigation is clean and new team members pick it up quickly.",
                "The UX feels modern and thoughtfully designed.",
                "Finding settings and tools is straightforward.",
            ],
            "es": [
                "La interfaz es muy intuitiva, pude configurar todo en minutos.",
                "La navegación es clara y los nuevos miembros aprenden rápido.",
                "La experiencia de usuario se siente moderna y bien pensada.",
                "Encontrar ajustes y herramientas es sencillo.",
            ],
        },
        "neutral": {
            "en": [
                "The UI is usable but some menus feel cluttered.",
                "It works once you learn where things are, but there's a learning curve.",
                "Design is fine overall, though a few flows feel outdated.",
                "Usability is decent; some actions require too many clicks.",
            ],
            "es": [
                "La interfaz es usable pero algunos menús se sienten saturados.",
                "Funciona cuando aprendes dónde está todo, pero hay curva de aprendizaje.",
                "El diseño está bien en general, aunque algunos flujos se sienten antiguos.",
                "La usabilidad es decente; algunas acciones requieren demasiados clics.",
            ],
        },
        "negative": {
            "en": [
                "The interface is confusing and our team struggles daily.",
                "Important actions are buried under too many nested menus.",
                "The mobile experience is nearly unusable for our field staff.",
                "We keep clicking the wrong buttons because labels are unclear.",
            ],
            "es": [
                "La interfaz es confusa y nuestro equipo lucha con ella a diario.",
                "Las acciones importantes están enterradas bajo demasiados menús.",
                "La experiencia móvil es casi inutilizable para nuestro equipo de campo.",
                "Seguimos pulsando botones incorrectos porque las etiquetas no son claras.",
            ],
        },
    },
    "support": {
        "positive": {
            "en": [
                "Support responded within an hour and solved our issue completely.",
                "The help center articles resolved my question on the first try.",
                "Our account manager has been proactive and very helpful.",
                "Live chat support was knowledgeable and friendly.",
            ],
            "es": [
                "El soporte respondió en menos de una hora y resolvió nuestro problema.",
                "Los artículos del centro de ayuda resolvieron mi duda a la primera.",
                "Nuestro account manager ha sido proactivo y muy útil.",
                "El chat en vivo fue competente y amable.",
            ],
        },
        "neutral": {
            "en": [
                "Support eventually helped, but it took longer than expected.",
                "The help docs are useful though sometimes out of date.",
                "We got an answer, but had to follow up twice.",
                "Support quality varies depending on who handles the ticket.",
            ],
            "es": [
                "El soporte ayudó al final, pero tardó más de lo esperado.",
                "La documentación es útil aunque a veces está desactualizada.",
                "Obtuvimos respuesta, pero tuvimos que insistir dos veces.",
                "La calidad del soporte varía según quién atienda el ticket.",
            ],
        },
        "negative": {
            "en": [
                "We've waited over a week with no response from support.",
                "Every support interaction feels like starting from scratch.",
                "The team closed our ticket without actually fixing the problem.",
                "Support keeps redirecting us without offering real solutions.",
            ],
            "es": [
                "Llevamos más de una semana sin respuesta del soporte.",
                "Cada interacción con soporte parece empezar desde cero.",
                "Cerraron nuestro ticket sin resolver el problema de verdad.",
                "El soporte nos redirige sin ofrecer soluciones reales.",
            ],
        },
    },
    "pricing": {
        "positive": {
            "en": [
                "The pricing is fair for the value we get from the platform.",
                "We find the Pro plan offers excellent ROI for our team size.",
                "Transparent pricing with no hidden fees is a big plus.",
                "The annual discount made upgrading an easy decision.",
            ],
            "es": [
                "El precio es justo por el valor que obtenemos de la plataforma.",
                "El plan Pro ofrece un excelente retorno para el tamaño de nuestro equipo.",
                "Precios transparentes sin cargos ocultos es una gran ventaja.",
                "El descuento anual hizo muy fácil la decisión de actualizar.",
            ],
        },
        "neutral": {
            "en": [
                "Pricing is competitive but the jump between tiers feels steep.",
                "The cost is reasonable, though we'd like more mid-tier options.",
                "We understand the pricing model but wish add-ons were clearer.",
                "Price is okay for now; we'll reassess at renewal.",
            ],
            "es": [
                "El precio es competitivo pero el salto entre planes se siente alto.",
                "El costo es razonable, aunque nos gustarían más opciones intermedias.",
                "Entendemos el modelo de precios pero los add-ons podrían ser más claros.",
                "El precio está bien por ahora; reevaluaremos en la renovación.",
            ],
        },
        "negative": {
            "en": [
                "The price increased 40% at renewal with no added value.",
                "Essential features are locked behind the most expensive tier.",
                "We're paying enterprise rates but getting basic-level service.",
                "Competitors offer similar features at half the cost.",
            ],
            "es": [
                "El precio subió 40% en la renovación sin valor agregado.",
                "Funciones esenciales están bloqueadas en el plan más caro.",
                "Pagamos tarifas enterprise pero recibimos un servicio básico.",
                "Competidores ofrecen funciones similares a la mitad del costo.",
            ],
        },
    },
    "features": {
        "positive": {
            "en": [
                "The automation features save our team several hours each week.",
                "Recent feature releases align well with what we actually need.",
                "Custom workflows are powerful and flexible.",
                "The analytics module exceeded our expectations.",
            ],
            "es": [
                "Las funciones de automatización nos ahorran varias horas cada semana.",
                "Las nuevas funciones encajan bien con lo que realmente necesitamos.",
                "Los flujos personalizados son potentes y flexibles.",
                "El módulo de analítica superó nuestras expectativas.",
            ],
        },
        "neutral": {
            "en": [
                "Core features work well, but advanced options feel limited.",
                "The feature set covers basics; we'd like more customization.",
                "Some promised roadmap items still haven't shipped.",
                "Features are adequate for now but not a differentiator.",
            ],
            "es": [
                "Las funciones principales funcionan bien, pero las avanzadas son limitadas.",
                "El conjunto de funciones cubre lo básico; queremos más personalización.",
                "Algunos ítems del roadmap prometido aún no se han lanzado.",
                "Las funciones son adecuadas por ahora pero no nos diferencian.",
            ],
        },
        "negative": {
            "en": [
                "Key features advertised on the website simply don't exist yet.",
                "The reporting module lacks basic filters we need daily.",
                "We can't use half the features because they're buggy.",
                "Competitors shipped these features months ago.",
            ],
            "es": [
                "Funciones clave anunciadas en la web simplemente no existen aún.",
                "El módulo de reportes carece de filtros básicos que usamos a diario.",
                "No podemos usar la mitad de las funciones porque tienen errores.",
                "Los competidores lanzaron estas funciones hace meses.",
            ],
        },
    },
    "onboarding": {
        "positive": {
            "en": [
                "Onboarding was smooth and our team was productive within a day.",
                "The guided setup wizard made configuration effortless.",
                "Training materials helped us adopt the platform quickly.",
                "We felt supported from signup through first successful workflow.",
            ],
            "es": [
                "La incorporación fue fluida y el equipo fue productivo en un día.",
                "El asistente de configuración hizo todo muy sencillo.",
                "Los materiales de capacitación nos ayudaron a adoptar la plataforma rápido.",
                "Nos sentimos acompañados desde el registro hasta el primer flujo exitoso.",
            ],
        },
        "neutral": {
            "en": [
                "Onboarding was fine but took longer than the sales demo suggested.",
                "Setup steps are clear, though some defaults needed manual tweaking.",
                "We got started eventually, but documentation could be better organized.",
                "The onboarding call helped, but follow-up resources were sparse.",
            ],
            "es": [
                "La incorporación estuvo bien pero tardó más de lo que prometió la demo.",
                "Los pasos de configuración son claros, aunque algunos valores por defecto requirieron ajustes.",
                "Arrancamos al final, pero la documentación podría estar mejor organizada.",
                "La llamada de onboarding ayudó, pero los recursos de seguimiento fueron escasos.",
            ],
        },
        "negative": {
            "en": [
                "We were left alone after signup with no clear next steps.",
                "Onboarding documentation is outdated and contradicts the actual UI.",
                "Our team gave up on setup after hitting multiple dead ends.",
                "Promised onboarding support never materialized.",
            ],
            "es": [
                "Nos dejaron solos tras el registro sin pasos claros a seguir.",
                "La documentación de onboarding está desactualizada y contradice la interfaz.",
                "El equipo abandonó la configuración tras varios callejones sin salida.",
                "El soporte de incorporación prometido nunca llegó.",
            ],
        },
    },
    "exports": {
        "positive": {
            "en": [
                "CSV exports are fast and include all the fields we need.",
                "Scheduled exports to our data warehouse work flawlessly.",
                "Export formats are flexible and easy to automate.",
                "We rely on daily exports and they've been 100% reliable.",
            ],
            "es": [
                "Las exportaciones CSV son rápidas e incluyen todos los campos que necesitamos.",
                "Las exportaciones programadas a nuestro data warehouse funcionan sin fallos.",
                "Los formatos de exportación son flexibles y fáciles de automatizar.",
                "Dependemos de exportaciones diarias y han sido 100% confiables.",
            ],
        },
        "neutral": {
            "en": [
                "Exports work but column naming could be more consistent.",
                "PDF exports are readable though formatting could improve.",
                "We can export data, but large exports sometimes time out.",
                "Export options cover most needs with a few gaps.",
            ],
            "es": [
                "Las exportaciones funcionan pero los nombres de columnas podrían ser más consistentes.",
                "Las exportaciones PDF son legibles aunque el formato podría mejorar.",
                "Podemos exportar datos, pero las exportaciones grandes a veces fallan por tiempo.",
                "Las opciones de exportación cubren la mayoría de necesidades con algunas brechas.",
            ],
        },
        "negative": {
            "en": [
                "Exported files are missing critical columns we need for compliance.",
                "Export jobs fail silently and we only discover it days later.",
                "We can't export historical data beyond 30 days on our plan.",
                "CSV exports corrupt special characters in our international data.",
            ],
            "es": [
                "Los archivos exportados omiten columnas críticas que necesitamos para cumplimiento.",
                "Las exportaciones fallan en silencio y solo lo descubrimos días después.",
                "No podemos exportar datos históricos más allá de 30 días en nuestro plan.",
                "Las exportaciones CSV corrompen caracteres especiales en nuestros datos internacionales.",
            ],
        },
    },
    "integrations": {
        "positive": {
            "en": [
                "The Slack integration keeps our team in sync without context switching.",
                "Salesforce sync works reliably and saved us manual data entry.",
                "API documentation is clear and webhooks fire consistently.",
                "Zapier integration connected our stack in under an hour.",
            ],
            "es": [
                "La integración con Slack mantiene al equipo sincronizado sin cambiar de contexto.",
                "La sincronización con Salesforce funciona bien y eliminó entrada manual de datos.",
                "La documentación de la API es clara y los webhooks funcionan de forma consistente.",
                "La integración con Zapier conectó nuestro stack en menos de una hora.",
            ],
        },
        "neutral": {
            "en": [
                "Integrations work for common tools but niche connectors are missing.",
                "The API is functional though rate limits can be restrictive.",
                "Webhook setup required extra troubleshooting on our end.",
                "Most integrations are stable; a few need periodic reauthorization.",
            ],
            "es": [
                "Las integraciones funcionan para herramientas comunes pero faltan conectores de nicho.",
                "La API es funcional aunque los límites de tasa pueden ser restrictivos.",
                "Configurar webhooks requirió troubleshooting adicional de nuestro lado.",
                "La mayoría de integraciones son estables; algunas requieren reautorización periódica.",
            ],
        },
        "negative": {
            "en": [
                "Our CRM integration broke after the latest release and data is out of sync.",
                "The API returns inconsistent errors with no useful documentation.",
                "Webhook deliveries fail randomly and we lose critical events.",
                "We can't connect our internal tools because OAuth keeps failing.",
            ],
            "es": [
                "Nuestra integración con el CRM se rompió tras el último release y los datos están desincronizados.",
                "La API devuelve errores inconsistentes sin documentación útil.",
                "Las entregas por webhook fallan al azar y perdemos eventos críticos.",
                "No podemos conectar nuestras herramientas internas porque OAuth falla constantemente.",
            ],
        },
    },
    "account_access": {
        "positive": {
            "en": [
                "SSO login works seamlessly across our entire organization.",
                "Role-based permissions give us the control we need.",
                "Account recovery was quick and secure.",
                "Multi-factor authentication setup was straightforward.",
            ],
            "es": [
                "El inicio de sesión SSO funciona sin problemas en toda la organización.",
                "Los permisos por rol nos dan el control que necesitamos.",
                "La recuperación de cuenta fue rápida y segura.",
                "Configurar la autenticación multifactor fue sencillo.",
            ],
        },
        "neutral": {
            "en": [
                "Access controls work but admin settings are hard to find.",
                "Login is reliable, though password reset emails are slow.",
                "Permissions are adequate but lack fine-grained options.",
                "SSO works most of the time with occasional hiccups.",
            ],
            "es": [
                "Los controles de acceso funcionan pero los ajustes de admin son difíciles de encontrar.",
                "El login es confiable, aunque los correos de restablecimiento tardan.",
                "Los permisos son adecuados pero carecen de opciones granulares.",
                "El SSO funciona la mayoría del tiempo con fallos ocasionales.",
            ],
        },
        "negative": {
            "en": [
                "Our entire team was locked out for two days with no explanation.",
                "Admin accounts lost permissions after the last security update.",
                "I can't reset my password and support hasn't helped in days.",
                "Unauthorized access to our workspace was not detected promptly.",
            ],
            "es": [
                "Todo el equipo quedó bloqueado dos días sin explicación.",
                "Las cuentas de admin perdieron permisos tras la última actualización de seguridad.",
                "No puedo restablecer mi contraseña y el soporte no ha ayudado en días.",
                "Un acceso no autorizado a nuestro workspace no fue detectado a tiempo.",
            ],
        },
    },
}


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    items = list(weights.keys())
    probs = [weights[item] for item in items]
    return rng.choices(items, weights=probs, k=1)[0]


def _sample_sentiment_for_channel(rng: random.Random, channel: str) -> str:
    return _weighted_choice(rng, CHANNEL_SENTIMENT_WEIGHTS[channel])


def _sample_product_area(rng: random.Random) -> str:
    return rng.choice(PRODUCT_AREAS)


def _sample_priority(rng: random.Random, product_area: str, sentiment: str) -> str:
    if product_area in HIGH_PRIORITY_AREAS:
        weights = PRIORITY_AREA_WEIGHTS["high_risk"].copy()
    else:
        weights = PRIORITY_AREA_WEIGHTS["default"].copy()

    if sentiment == "negative":
        weights["low"] *= 0.5
        weights["medium"] *= 0.8
        weights["high"] *= 1.3
        weights["critical"] *= 1.5
    elif sentiment == "positive":
        weights["low"] *= 1.5
        weights["medium"] *= 1.2
        weights["high"] *= 0.7
        weights["critical"] *= 0.3

    priority = _weighted_choice(rng, weights)

    if priority == "critical" and product_area not in CRITICAL_PRIORITY_AREAS:
        if rng.random() < 0.75:
            priority = _weighted_choice(
                rng,
                {"low": 0.2, "medium": 0.35, "high": 0.45, "critical": 0.0},
            )

    return priority


def _sample_rating(rng: random.Random, sentiment: str) -> int:
    if sentiment == "positive":
        return rng.choices([4, 5], weights=[0.35, 0.65], k=1)[0]
    if sentiment == "neutral":
        return 3
    return rng.choices([1, 2], weights=[0.55, 0.45], k=1)[0]


def _sample_response_time_hours(rng: random.Random, priority: str) -> float:
    low, high = RESPONSE_TIME_RANGES[priority]
    return round(rng.uniform(low, high), 1)


def _sample_resolved(rng: random.Random, priority: str, sentiment: str) -> bool:
    base_prob = {
        "low": 0.92,
        "medium": 0.82,
        "high": 0.62,
        "critical": 0.38,
    }[priority]

    if sentiment == "positive":
        base_prob += 0.05
    elif sentiment == "neutral":
        base_prob += 0.02
    else:
        base_prob -= 0.12

    if priority == "critical" and sentiment == "negative":
        base_prob -= 0.15

    base_prob = max(0.05, min(0.98, base_prob))
    return rng.random() < base_prob


def _build_message(
    rng: random.Random,
    product_area: str,
    sentiment: str,
    language: str,
    customer_segment: str,
    plan_type: str,
) -> str:
    templates = MESSAGE_TEMPLATES[product_area][sentiment][language]
    message = rng.choice(templates)

    if rng.random() < 0.45:
        suffix_map = SEGMENT_SUFFIXES_EN if language == "en" else SEGMENT_SUFFIXES_ES
        message = f"{message} {suffix_map[customer_segment]}"
    elif rng.random() < 0.35:
        plan_map = PLAN_SUFFIXES_EN if language == "en" else PLAN_SUFFIXES_ES
        message = f"{message} {plan_map[plan_type]}"

    return message


def _random_date(rng: random.Random) -> str:
    start = datetime(2025, 1, 1)
    end = datetime(2025, 12, 31)
    delta_days = (end - start).days
    random_day = start + timedelta(days=rng.randint(0, delta_days))
    return random_day.strftime("%Y-%m-%d")


def generate_feedback_dataset(n: int = 1000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    records: list[dict] = []

    for i in range(1, n + 1):
        channel = rng.choice(CHANNELS)
        sentiment = _sample_sentiment_for_channel(rng, channel)
        product_area = _sample_product_area(rng)
        priority = _sample_priority(rng, product_area, sentiment)
        language = "en" if np_rng.random() < 0.60 else "es"
        customer_segment = rng.choice(CUSTOMER_SEGMENTS)
        plan_type = rng.choice(PLAN_TYPES)

        record = {
            "feedback_id": f"FB-{i:05d}",
            "date": _random_date(rng),
            "channel": channel,
            "customer_segment": customer_segment,
            "plan_type": plan_type,
            "language": language,
            "message": _build_message(
                rng, product_area, sentiment, language, customer_segment, plan_type
            ),
            "rating": _sample_rating(rng, sentiment),
            "product_area": product_area,
            "sentiment_label": sentiment,
            "priority": priority,
            "resolved": _sample_resolved(rng, priority, sentiment),
            "response_time_hours": _sample_response_time_hours(rng, priority),
        }
        records.append(record)

    return pd.DataFrame(records)


def save_dataset(df: pd.DataFrame, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def print_summary(df: pd.DataFrame) -> None:
    print(f"Row count: {len(df)}")
    print("\nSentiment distribution:")
    print(df["sentiment_label"].value_counts().to_string())
    print("\nChannel distribution:")
    print(df["channel"].value_counts().to_string())
    print("\nPriority distribution:")
    print(df["priority"].value_counts().to_string())
    print("\nLanguage split:")
    print(df["language"].value_counts().to_string())
    print(f"\nOutput path: data/raw/customer_feedback_raw.csv")


def main() -> None:
    df = generate_feedback_dataset()
    output_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "customer_feedback_raw.csv"
    save_dataset(df, str(output_path))
    print_summary(df)


if __name__ == "__main__":
    main()
