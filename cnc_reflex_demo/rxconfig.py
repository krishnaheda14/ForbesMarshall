import reflex as rx

config = rx.Config(
    app_name="cnc_reflex_demo",
    db_url="sqlite:///reflex.db",
    env=rx.Env.DEV,
)
