"""
CNC Job Scheduling System - Reflex UI
Production-ready web app built with Reflex (latest syntax)
"""

import reflex as rx
from typing import List, Dict
import pandas as pd
import sys
import os

# Add parent directory to import existing scheduling code
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ScheduleState(rx.State):
    """Application state for CNC scheduling"""
    
    # Current selections
    selected_heuristic: str = "SPT"
    hourly_rate: float = 30.0
    cost_threshold: float = 0.85
    
    # Loading states
    is_computing: bool = False
    is_loading_data: bool = False
    
    # Metrics
    makespan_days: float = 0.0
    total_tardiness: float = 0.0
    utilization_pct: float = 0.0
    total_cost: float = 0.0
    on_time_pct: float = 0.0
    late_operations: int = 0
    
    # Comparison results
    comparison_results: List[Dict] = []
    has_comparison: bool = False
    
    # Messages
    status_message: str = "Ready to compute schedule"
    status_type: str = "info"  # info, success, warning, error
    
    def set_heuristic(self, value: str):
        """Update selected heuristic"""
        self.selected_heuristic = value
        self.status_message = f"Selected {value} algorithm"
        self.status_type = "info"
    
    def set_hourly_rate(self, value: str):
        """Update hourly rate"""
        try:
            self.hourly_rate = float(value)
        except:
            pass
    
    def set_cost_threshold(self, value: str):
        """Update cost threshold"""
        try:
            self.cost_threshold = float(value)
        except:
            pass
    
    async def compute_schedule(self):
        """Compute schedule using selected heuristic"""
        self.is_computing = True
        self.status_message = f"Computing {self.selected_heuristic} schedule..."
        self.status_type = "info"
        yield
        
        try:
            # Import scheduling functions
            from cnc_scheduling import run_single_heuristic, calculate_metrics, load_all_data
            
            # Load data
            df_ops, df_machines, df_effective, df_penalties, df_vendors = load_all_data()
            
            # Run scheduler
            schedule = run_single_heuristic(
                df_ops.copy(),
                df_machines.copy(),
                df_effective.copy(),
                df_penalties.copy(),
                heuristic=self.selected_heuristic
            )
            
            # Calculate metrics
            metrics = calculate_metrics(
                schedule,
                df_ops,
                self.selected_heuristic,
                hourly_rate=self.hourly_rate
            )
            
            # Update state with results
            self.makespan_days = metrics.get('makespan_days', 0.0)
            self.total_tardiness = metrics.get('total_tardiness_days', 0.0)
            self.utilization_pct = metrics.get('machine_utilization', 0.0)
            self.total_cost = metrics.get('total_cost', 0.0)
            self.on_time_pct = metrics.get('on_time_pct', 0.0)
            self.late_operations = metrics.get('late_operations', 0)
            
            self.status_message = f"✅ {self.selected_heuristic} schedule computed successfully!"
            self.status_type = "success"
            
        except Exception as e:
            self.status_message = f"❌ Error: {str(e)}"
            self.status_type = "error"
            
        finally:
            self.is_computing = False
            yield
    
    async def compute_all_heuristics(self):
        """Compare all scheduling heuristics"""
        self.is_computing = True
        self.status_message = "Computing all heuristics for comparison..."
        self.status_type = "info"
        yield
        
        try:
            from cnc_scheduling import run_single_heuristic, calculate_metrics, load_all_data
            
            df_ops, df_machines, df_effective, df_penalties, df_vendors = load_all_data()
            
            heuristics = ['SPT', 'EDD', 'CR', 'PRIORITY', 'WEIGHTED', 'SLACK']
            results = []
            
            for heuristic in heuristics:
                schedule = run_single_heuristic(
                    df_ops.copy(),
                    df_machines.copy(),
                    df_effective.copy(),
                    df_penalties.copy(),
                    heuristic=heuristic
                )
                
                metrics = calculate_metrics(schedule, df_ops, heuristic, self.hourly_rate)
                
                results.append({
                    'heuristic': heuristic,
                    'makespan': metrics.get('makespan_days', 0.0),
                    'tardiness': metrics.get('total_tardiness_days', 0.0),
                    'utilization': metrics.get('machine_utilization', 0.0),
                    'cost': metrics.get('total_cost', 0.0),
                    'on_time': metrics.get('on_time_pct', 0.0)
                })
            
            self.comparison_results = results
            self.has_comparison = True
            self.status_message = "✅ All heuristics compared successfully!"
            self.status_type = "success"
            
        except Exception as e:
            self.status_message = f"❌ Error: {str(e)}"
            self.status_type = "error"
            
        finally:
            self.is_computing = False
            yield


def hero_section() -> rx.Component:
    """Hero banner with gradient background"""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.text("🏭", font_size="3rem"),
                rx.heading(
                    "CNC Job Scheduling System",
                    size="9",
                    color="white",
                    weight="bold"
                ),
                spacing="3",
                align="center"
            ),
            rx.text(
                "AI-powered optimization for manufacturing excellence",
                size="5",
                color="rgba(255,255,255,0.9)",
                text_align="center"
            ),
            spacing="3",
            align="center"
        ),
        background="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        padding="3rem",
        border_radius="12px",
        margin_bottom="2rem",
        box_shadow="0 10px 40px rgba(0,0,0,0.15)"
    )


def metric_card(label: str, value: str, icon: str, color: str = "blue") -> rx.Component:
    """Professional metric card component"""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.text(icon, font_size="1.5rem"),
                rx.text(
                    label,
                    size="2",
                    color="gray",
                    weight="medium",
                    text_transform="uppercase"
                ),
                spacing="2",
                align="center"
            ),
            rx.text(
                value,
                size="8",
                weight="bold",
                color=color
            ),
            spacing="2",
            align="start"
        ),
        style={
            "border_left": f"4px solid var(--{color}-9)",
            "transition": "transform 0.2s ease",
            "_hover": {
                "transform": "translateY(-4px)",
                "box_shadow": "0 8px 30px rgba(0,0,0,0.12)"
            }
        }
    )


def control_panel() -> rx.Component:
    """Scheduling controls panel"""
    return rx.card(
        rx.vstack(
            rx.heading("Schedule Controls", size="6", weight="bold"),
            
            rx.grid(
                # Heuristic selector
                rx.vstack(
                    rx.text("Algorithm", size="2", weight="medium", color="gray"),
                    rx.select(
                        ["SPT", "EDD", "CR", "PRIORITY", "WEIGHTED", "SLACK"],
                        value=ScheduleState.selected_heuristic,
                        on_change=ScheduleState.set_heuristic,
                        size="3"
                    ),
                    spacing="2",
                    align="start",
                    width="100%"
                ),
                
                # Hourly rate
                rx.vstack(
                    rx.text("Hourly Rate ($)", size="2", weight="medium", color="gray"),
                    rx.input(
                        value=ScheduleState.hourly_rate,
                        on_change=ScheduleState.set_hourly_rate,
                        type="number",
                        size="3"
                    ),
                    spacing="2",
                    align="start",
                    width="100%"
                ),
                
                # Cost threshold
                rx.vstack(
                    rx.text("Cost Threshold", size="2", weight="medium", color="gray"),
                    rx.input(
                        value=ScheduleState.cost_threshold,
                        on_change=ScheduleState.set_cost_threshold,
                        type="number",
                        step="0.05",
                        size="3"
                    ),
                    spacing="2",
                    align="start",
                    width="100%"
                ),
                
                columns="3",
                spacing="4",
                width="100%"
            ),
            
            rx.hstack(
                rx.button(
                    "🧪 Compute Schedule",
                    on_click=ScheduleState.compute_schedule,
                    loading=ScheduleState.is_computing,
                    size="3",
                    color_scheme="blue",
                    variant="solid"
                ),
                rx.button(
                    "📊 Compare All",
                    on_click=ScheduleState.compute_all_heuristics,
                    loading=ScheduleState.is_computing,
                    size="3",
                    color_scheme="grass",
                    variant="outline"
                ),
                spacing="3",
                width="100%"
            ),
            
            spacing="4",
            width="100%"
        )
    )


def status_message() -> rx.Component:
    """Status message display"""
    return rx.cond(
        ScheduleState.status_message != "",
        rx.callout(
            ScheduleState.status_message,
            icon=rx.cond(
                ScheduleState.status_type == "success", "check",
                rx.cond(ScheduleState.status_type == "error", "x", "info")
            ),
            color_scheme=rx.cond(
                ScheduleState.status_type == "success", "green",
                rx.cond(ScheduleState.status_type == "error", "red", "blue")
            ),
            size="2"
        )
    )


def metrics_dashboard() -> rx.Component:
    """KPI metrics dashboard"""
    return rx.cond(
        ScheduleState.makespan_days > 0,
        rx.vstack(
            rx.heading("Performance Metrics", size="6", weight="bold"),
            
            rx.grid(
                metric_card(
                    "Makespan",
                    rx.text(f"{ScheduleState.makespan_days:.1f} days"),
                    "⏱️",
                    "blue"
                ),
                metric_card(
                    "Tardiness",
                    rx.text(f"{ScheduleState.total_tardiness:.1f} days"),
                    "⚠️",
                    "amber"
                ),
                metric_card(
                    "Utilization",
                    rx.text(f"{ScheduleState.utilization_pct:.1f}%"),
                    "📊",
                    "grass"
                ),
                metric_card(
                    "Total Cost",
                    rx.text(f"${ScheduleState.total_cost:,.0f}"),
                    "💰",
                    "purple"
                ),
                columns="4",
                spacing="4",
                width="100%"
            ),
            
            spacing="4",
            width="100%"
        )
    )


def comparison_table() -> rx.Component:
    """Heuristic comparison table"""
    return rx.cond(
        ScheduleState.has_comparison,
        rx.vstack(
            rx.heading("Heuristic Comparison", size="6", weight="bold"),
            
            rx.card(
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Algorithm"),
                            rx.table.column_header_cell("Makespan"),
                            rx.table.column_header_cell("Tardiness"),
                            rx.table.column_header_cell("Utilization"),
                            rx.table.column_header_cell("Cost"),
                            rx.table.column_header_cell("On-Time %")
                        )
                    ),
                    rx.table.body(
                        rx.foreach(
                            ScheduleState.comparison_results,
                            lambda result: rx.table.row(
                                rx.table.cell(
                                    rx.badge(result["heuristic"], color_scheme="blue")
                                ),
                                rx.table.cell(f"{result['makespan']:.1f} days"),
                                rx.table.cell(f"{result['tardiness']:.1f} days"),
                                rx.table.cell(f"{result['utilization']:.1f}%"),
                                rx.table.cell(f"${result['cost']:,.0f}"),
                                rx.table.cell(f"{result['on_time']:.1f}%")
                            )
                        )
                    ),
                    variant="surface",
                    size="3"
                )
            ),
            
            spacing="4",
            width="100%"
        )
    )


def algorithm_info() -> rx.Component:
    """Algorithm information cards"""
    return rx.grid(
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.text("🎯", font_size="2rem"),
                    rx.heading("SPT", size="5", weight="bold"),
                    spacing="2"
                ),
                rx.text(
                    "Shortest Processing Time",
                    size="2",
                    color="gray",
                    weight="medium"
                ),
                rx.text(
                    "Minimizes makespan and average flow time. Best for quick turnaround.",
                    size="2"
                ),
                rx.badge("Recommended", color_scheme="green"),
                spacing="3",
                align="start"
            )
        ),
        
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.text("📅", font_size="2rem"),
                    rx.heading("EDD", size="5", weight="bold"),
                    spacing="2"
                ),
                rx.text(
                    "Earliest Due Date",
                    size="2",
                    color="gray",
                    weight="medium"
                ),
                rx.text(
                    "Prioritizes urgent jobs to minimize tardiness. Best for deadline-critical work.",
                    size="2"
                ),
                rx.badge("Deadline-Focused", color_scheme="blue"),
                spacing="3",
                align="start"
            )
        ),
        
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.text("⚖️", font_size="2rem"),
                    rx.heading("CR", size="5", weight="bold"),
                    spacing="2"
                ),
                rx.text(
                    "Critical Ratio",
                    size="2",
                    color="gray",
                    weight="medium"
                ),
                rx.text(
                    "Balances urgency with work remaining. Best for mixed workloads.",
                    size="2"
                ),
                rx.badge("Balanced", color_scheme="purple"),
                spacing="3",
                align="start"
            )
        ),
        
        columns="3",
        spacing="4",
        width="100%"
    )


def index() -> rx.Component:
    """Main page layout"""
    return rx.container(
        rx.vstack(
            hero_section(),
            status_message(),
            control_panel(),
            metrics_dashboard(),
            comparison_table(),
            
            rx.heading("Available Algorithms", size="6", weight="bold", margin_top="2rem"),
            algorithm_info(),
            
            spacing="6",
            width="100%"
        ),
        size="4",
        padding="2rem"
    )


# Create app
app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="blue"
    )
)

app.add_page(
    index,
    title="CNC Job Scheduling System",
    description="AI-powered manufacturing optimization platform"
)
