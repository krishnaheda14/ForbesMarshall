"""CNC Scheduler - Production UI with Reflex"""
import reflex as rx

class State(rx.State):
    """Application state"""
    heuristic: str = "SPT"
    is_computing: bool = False
    makespan: float = 14.2
    tardiness: float = 2.5
    utilization: float = 87.3
    total_cost: float = 45230
    
    def set_heuristic(self, value: str):
        self.heuristic = value
    
    def compute(self):
        self.is_computing = True
        yield
        # Simulate computation
        if self.heuristic == "SPT":
            self.makespan = 14.2
            self.tardiness = 2.5
            self.utilization = 87.3
        elif self.heuristic == "EDD":
            self.makespan = 15.1
            self.tardiness = 1.8
            self.utilization = 82.5
        elif self.heuristic == "CR":
            self.makespan = 14.8
            self.tardiness = 2.1
            self.utilization = 85.0
        self.is_computing = False

def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            # Header
            rx.box(
                rx.vstack(
                    rx.heading("🏭 CNC Job Scheduling System", size="9", color="white"),
                    rx.text("Production-ready UI built with Reflex", size="4", color="rgba(255,255,255,0.9)"),
                    spacing="2",
                ),
                background="linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)",
                padding="3rem 2rem",
                border_radius="16px",
                margin_bottom="2rem",
            ),
            
            # Metrics Grid
            rx.grid(
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("clock", size=24, color="blue"),
                            rx.text("MAKESPAN", size="2", color="gray", weight="bold"),
                            spacing="2",
                        ),
                        rx.heading(f"{State.makespan:.1f} days", size="8"),
                        rx.hstack(
                            rx.icon("trending-down", size=16, color="green"),
                            rx.text("-12%", size="2", color="green", weight="bold"),
                            spacing="1",
                        ),
                        spacing="2",
                        align="start",
                    ),
                ),
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("alert-triangle", size=24, color="amber"),
                            rx.text("TARDINESS", size="2", color="gray", weight="bold"),
                            spacing="2",
                        ),
                        rx.heading(f"{State.tardiness:.1f} days", size="8"),
                        rx.hstack(
                            rx.icon("trending-down", size=16, color="green"),
                            rx.text("-8%", size="2", color="green", weight="bold"),
                            spacing="1",
                        ),
                        spacing="2",
                        align="start",
                    ),
                ),
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("trending-up", size=24, color="green"),
                            rx.text("UTILIZATION", size="2", color="gray", weight="bold"),
                            spacing="2",
                        ),
                        rx.heading(f"{State.utilization:.1f}%", size="8"),
                        rx.hstack(
                            rx.icon("trending-up", size=16, color="green"),
                            rx.text("+5.2%", size="2", color="green", weight="bold"),
                            spacing="1",
                        ),
                        spacing="2",
                        align="start",
                    ),
                ),
                rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.icon("dollar-sign", size=24, color="purple"),
                            rx.text("TOTAL COST", size="2", color="gray", weight="bold"),
                            spacing="2",
                        ),
                        rx.heading(f"${State.total_cost:,.0f}", size="8"),
                        rx.hstack(
                            rx.icon("trending-down", size=16, color="green"),
                            rx.text("-$3.2K", size="2", color="green", weight="bold"),
                            spacing="1",
                        ),
                        spacing="2",
                        align="start",
                    ),
                ),
                columns="4",
                spacing="4",
                margin_bottom="2rem",
            ),
            
            # Control Panel
            rx.card(
                rx.vstack(
                    rx.heading("Schedule Controls", size="6", weight="bold"),
                    rx.hstack(
                        rx.vstack(
                            rx.text("Algorithm", size="2", weight="medium", color="gray"),
                            rx.select(
                                ["SPT", "EDD", "CR", "PRIORITY", "WEIGHTED", "SLACK"],
                                value=State.heuristic,
                                on_change=State.set_heuristic,
                                size="3",
                            ),
                            align="start",
                            spacing="1",
                        ),
                        rx.vstack(
                            rx.text("Actions", size="2", weight="medium", color="gray"),
                            rx.button(
                                rx.icon("play", size=18),
                                "Compute Schedule",
                                on_click=State.compute,
                                loading=State.is_computing,
                                size="3",
                                color_scheme="blue",
                            ),
                            align="start",
                            spacing="1",
                        ),
                        spacing="6",
                    ),
                    spacing="4",
                    align="start",
                ),
            ),
            
            # Algorithm Info
            rx.card(
                rx.hstack(
                    rx.icon("zap", size=40, color="blue"),
                    rx.vstack(
                        rx.heading(rx.cond(
                            State.heuristic == "SPT", "Shortest Processing Time",
                            rx.cond(
                                State.heuristic == "EDD", "Earliest Due Date",
                                "Critical Ratio"
                            )
                        ), size="5", weight="bold"),
                        rx.text(rx.cond(
                            State.heuristic == "SPT", "Minimizes makespan and average flow time",
                            rx.cond(
                                State.heuristic == "EDD", "Minimizes tardiness and late deliveries",
                                "Balances urgency with work remaining"
                            )
                        ), size="2", color="gray"),
                        spacing="1",
                        align="start",
                    ),
                    spacing="4",
                ),
                background="linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(147, 197, 253, 0.05) 100%)",
                margin_top="2rem",
            ),
            
            # Footer
            rx.box(
                rx.vstack(
                    rx.divider(),
                    rx.text("🏭 ForbesMarshall CNC Scheduling System | v2.0 | Built with Reflex", 
                           size="2", color="gray", text_align="center"),
                    spacing="3",
                ),
                padding_top="3rem",
            ),
            
            spacing="6",
            padding="2rem 0",
        ),
        max_width="1400px",
    )

app = rx.App()
app.add_page(index, route="/", title="CNC Scheduler")
