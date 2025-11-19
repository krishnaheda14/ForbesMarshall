# Reflex App Configuration

import reflex as rx

config = rx.Config(
    app_name="simple_reflex",
    db_url="sqlite:///reflex.db",
    env=rx.Env.DEV,
)
