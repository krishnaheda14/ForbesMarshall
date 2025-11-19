"""
CNC Scheduler - Simplified Reflex Demo
Standalone version that works without complex folder structure
"""

import reflex as rx

# Simple state
class State(rx.State):
    heuristic: str = "SPT"
    is_computing: bool = False
    makespan: float = 14.2
    tardiness: float = 2.5
    utilization: float = 87.3
    
    def compute(self):
        self.is_computing = True
        yield
        # Simulate computation
        import time
        time.sleep(1)
        if self.heuristic == "SPT":
            self.makespan = 14.2
        elif self.heuristic == "EDD":
            self.makespan = 15.1
        self.is_computing = False

# Simple UI
def index():
    return rx.container(
        rx.vstack(
            # Header
            rx.heading("🏭 CNC Job Scheduling", size="9"),
            rx.text("Built with Reflex - Python-only web framework", size="4", color="gray"),
            
            # Metrics
            rx.hstack(
                rx.card(
                    rx.vstack(
                        rx.text("Makespan", size="2", color="gray"),
                        rx.heading(f"{State.makespan:.1f} days", size="7"),
                        spacing="2",
                    )
                ),
                rx.card(
                    rx.vstack(
                        rx.text("Tardiness", size="2", color="gray"),
                        rx.heading(f"{State.tardiness:.1f} days", size="7"),
                        spacing="2",
                    )
                ),
                rx.card(
                    rx.vstack(
                        rx.text("Utilization", size="2", color="gray"),
                        rx.heading(f"{State.utilization:.1f}%", size="7"),
                        spacing="2",
                    )
                ),
                spacing="4",
            ),
            
            # Controls
            rx.card(
                rx.vstack(
                    rx.heading("Controls", size="5"),
                    rx.select(
                        ["SPT", "EDD", "CR", "PRIORITY"],
                        value=State.heuristic,
                        on_change=State.set_heuristic,
                    ),
                    rx.button(
                        "Compute Schedule",
                        on_click=State.compute,
                        loading=State.is_computing,
                        size="3",
                    ),
                    spacing="3",
                )
            ),
            
            spacing="6",
            padding="4rem",
        ),
        max_width="1200px",
    )

# Create app
app = rx.App()
app.add_page(index)
