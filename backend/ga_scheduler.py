# backend/ga_scheduler.py
"""
Genetic Algorithm Scheduler for Job Shop Scheduling
High Explainability Implementation - Every decision is logged and explained
"""

import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import copy
import time
import logging
import concurrent.futures

from cnc_scheduler_core import (
    get_eligible_machines,
    calculate_inhouse_cost,
    make_or_buy_decision,
    get_setup_penalty,
    calculate_metrics
)

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s [GA] %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class Chromosome:
    """
    Represents one scheduling solution (chromosome)
    
    Gene Structure:
    - Each gene represents one operation
    - Gene value = (machine_id, position_in_sequence, outsource_decision)
    
    Example: For operation J101_Op1:
      Gene = ('M1', 2, 'IN_HOUSE') means:
        - Assign to machine M1
        - 3rd position in M1's queue
        - Process in-house (not outsourced)
    """
    
    def __init__(self, operations: pd.DataFrame, machines: pd.DataFrame, 
                 df_effective: pd.DataFrame, df_penalties: pd.DataFrame):
        self.operations = operations
        self.machines = machines
        self.df_effective = df_effective
        self.df_penalties = df_penalties
        
        # Chromosome genes: dict mapping operation_id -> (machine_id, sequence_pos, outsource)
        self.genes: Dict[str, Tuple[str, int, str]] = {}
        
        # Fitness metrics
        self.fitness_score = float('-inf')
        self.metrics = {}
        self.schedule_df = None
        
        # Explainability tracking
        self.gene_explanations = {}  # Why each gene was assigned
        self.constraint_violations = []
        self.fitness_breakdown = {}
        
    def initialize_random(self, cost_threshold=0.9):
        """
        Create a random but valid chromosome
        
        How it works:
        1. For each operation, pick a random eligible machine
        2. Decide make-or-buy based on cost threshold
        3. Assign random position in machine queue
        
        Explainability: Log why each operation got its assignment
        """
        self.genes = {}
        self.gene_explanations = {}
        
        for idx, op in self.operations.iterrows():
            op_id = op['Operation_ID']
            op_type = op['Op_Type']
            
            # Get eligible machines for this operation type
            eligible = get_eligible_machines(op_type)
            
            if not eligible or len(eligible) == 0:
                # Force outsource if no eligible machines
                self.genes[op_id] = ('OUTSOURCE', 0, 'OUTSOURCE')
                self.gene_explanations[op_id] = f"No in-house machines can handle {op_type}"
                continue
            
            # Check if outsourcing is beneficial
            outsource_decision = 'IN_HOUSE'
            if op.get('Outsource_Flag') == 'Y':
                decision = make_or_buy_decision(op, self.df_effective, cost_threshold)
                if decision and decision[0] == 'OUTSOURCE':
                    outsource_decision = 'OUTSOURCE'
                    self.genes[op_id] = ('OUTSOURCE', 0, 'OUTSOURCE')
                    self.gene_explanations[op_id] = f"Vendor cost ${decision[1]:.2f} is cheaper than in-house"
                    continue
            
            # Randomly pick an eligible machine
            selected_machine = random.choice(eligible)
            random_position = random.randint(0, 100)  # Will be sorted later
            
            self.genes[op_id] = (selected_machine, random_position, outsource_decision)
            self.gene_explanations[op_id] = (
                f"Randomly assigned to {selected_machine} (eligible: {', '.join(eligible)}). "
                f"Position {random_position} in queue."
            )
    
    def decode_to_schedule(self) -> pd.DataFrame:
        """
        Convert chromosome genes into actual schedule with start/end times
        
        This is where genes become reality:
        1. Group operations by machine
        2. Sort by sequence position (gene's position value)
        3. Schedule operations respecting:
           - Precedence (operation sequence within jobs)
           - Release times
           - Machine availability
           - Maintenance windows
        
        Returns: DataFrame with Start_Time, End_Time, Machine_ID for each operation
        """
        schedule = []
        machine_availability = {m: 0 for m in self.machines['Machine_ID']}
        machine_last_material = {m: None for m in self.machines['Machine_ID']}
        op_completion_times = {}
        
        # Group operations by assigned machine
        machine_queues = {}
        for op_id, (machine_id, position, outsource) in self.genes.items():
            if outsource == 'OUTSOURCE':
                continue
            if machine_id not in machine_queues:
                machine_queues[machine_id] = []
            machine_queues[machine_id].append((position, op_id))
        
        # Sort each machine's queue by position
        for machine_id in machine_queues:
            machine_queues[machine_id].sort(key=lambda x: x[0])
        
        # Schedule operations machine by machine, respecting precedence
        max_iterations = 1000
        iteration = 0
        scheduled_ops = set()
        
        while len(scheduled_ops) < len(self.operations) and iteration < max_iterations:
            iteration += 1
            made_progress = False
            
            for machine_id, queue in machine_queues.items():
                for position, op_id in queue:
                    if op_id in scheduled_ops:
                        continue
                    
                    op = self.operations[self.operations['Operation_ID'] == op_id].iloc[0]
                    
                    # Check precedence: all prior ops in same job must be done
                    job_ops = self.operations[self.operations['Job_ID'] == op['Job_ID']].sort_values('Op_Seq')
                    can_schedule = True
                    earliest_start = op.get('Release_Time_Min', 0)
                    
                    for _, pred in job_ops.iterrows():
                        if pred['Op_Seq'] < op['Op_Seq']:
                            if pred['Operation_ID'] not in op_completion_times:
                                can_schedule = False
                                break
                            else:
                                earliest_start = max(earliest_start, op_completion_times[pred['Operation_ID']])
                    
                    if not can_schedule:
                        continue
                    
                    # Get processing details
                    op_details = self.df_effective[
                        (self.df_effective['Operation_ID'] == op_id) &
                        (self.df_effective['Machine_ID'] == machine_id)
                    ]
                    
                    if len(op_details) == 0:
                        # Constraint violation - machine can't handle this op
                        self.constraint_violations.append(
                            f"Invalid assignment: {op_id} to {machine_id}"
                        )
                        scheduled_ops.add(op_id)
                        continue
                    
                    eff_time = op_details.iloc[0]['Effective_Proc_Time']
                    prev_material = machine_last_material.get(machine_id)
                    setup_penalty = get_setup_penalty(prev_material, op.get('Mat_Type'), self.df_penalties)
                    actual_setup = op.get('Setup_Time', 0) + setup_penalty
                    transfer = op.get('Transfer_Min', 0)
                    total_duration = actual_setup + eff_time + transfer
                    
                    # Calculate start time (considering machine availability)
                    start_time = max(machine_availability[machine_id], earliest_start)
                    end_time = start_time + total_duration
                    
                    # Record schedule
                    schedule.append({
                        'Operation_ID': op_id,
                        'Job_ID': op['Job_ID'],
                        'Machine_ID': machine_id,
                        'Start_Time': start_time,
                        'End_Time': end_time,
                        'Setup_Time': actual_setup,
                        'Proc_Time': eff_time,
                        'Transfer_Time': transfer,
                        'Due_Time': op.get('Due_Time_Min', 0),
                        'Tardiness': max(0, end_time - op.get('Due_Time_Min', 0)),
                        'Priority': int(op.get('Priority', 3)),
                        'Assignment_Type': 'IN_HOUSE',
                        'Outsource_Cost': 0
                    })
                    
                    machine_availability[machine_id] = end_time
                    machine_last_material[machine_id] = op.get('Mat_Type')
                    op_completion_times[op_id] = end_time
                    scheduled_ops.add(op_id)
                    made_progress = True
            
            if not made_progress:
                break
        
        # Handle outsourced operations
        for op_id, (machine_id, position, outsource) in self.genes.items():
            if outsource == 'OUTSOURCE' and op_id not in scheduled_ops:
                op = self.operations[self.operations['Operation_ID'] == op_id].iloc[0]
                outsource_time = op.get('Outsource_Time_Min', op.get('Total_Proc_Min', 0))
                release_time = op.get('Release_Time_Min', 0)
                
                outsource_cost = float(op.get('Outsource_Cost', 0) or 0)
                if outsource_cost <= 0.1:
                    inhouse_res = calculate_inhouse_cost(op, self.df_effective)
                    if inhouse_res and inhouse_res[0]:
                        outsource_cost = inhouse_res[0] * 1.2
                    else:
                        outsource_cost = (float(op.get('Total_Proc_Min', 60)) / 60 * 50.0) + 50
                
                schedule.append({
                    'Operation_ID': op_id,
                    'Job_ID': op['Job_ID'],
                    'Machine_ID': 'OUTSOURCE',
                    'Start_Time': release_time,
                    'End_Time': release_time + outsource_time,
                    'Setup_Time': 0,
                    'Proc_Time': 0,
                    'Transfer_Time': 0,
                    'Due_Time': op.get('Due_Time_Min', 0),
                    'Tardiness': max(0, release_time + outsource_time - op.get('Due_Time_Min', 0)),
                    'Priority': int(op.get('Priority', 3)),
                    'Assignment_Type': 'OUTSOURCE',
                    'Outsource_Cost': outsource_cost
                })
        
        return pd.DataFrame(schedule)
    
    def calculate_fitness(self, weights: Dict[str, float] = None):
        """
        Calculate fitness score for this chromosome
        
        Fitness considers multiple objectives:
        - Minimize makespan (total time)
        - Minimize tardiness (late deliveries)
        - Minimize cost (labor + outsourcing)
        - Maximize utilization (don't waste capacity)
        
        Explainability: Break down exactly how fitness is calculated
        """
        if weights is None:
            weights = {
                'makespan': 0.25,
                'tardiness': 0.30,
                'cost': 0.25,
                'utilization': 0.20
            }
        
        # Decode genes to schedule
        self.schedule_df = self.decode_to_schedule()
        
        if self.schedule_df.empty:
            self.fitness_score = float('-inf')
            self.fitness_breakdown = {'error': 'Empty schedule'}
            return self.fitness_score
        
        # Merge with operation details for metrics
        # Be defensive: ensure required columns exist in operations dataframe
        ops_for_merge = self.operations.copy()
        required_cols = {
            'Priority': 3,
            'Total_Proc_Min': 0,
            'Release_Time_Min': 0,
            'Due_Time_Min': 0
        }
        for col, default in required_cols.items():
            if col not in ops_for_merge.columns:
                ops_for_merge[col] = default

        schedule_with_details = self.schedule_df.merge(
            ops_for_merge[['Operation_ID', 'Priority', 'Total_Proc_Min', 'Release_Time_Min', 'Due_Time_Min']],
            on='Operation_ID', how='left'
        )
        # Ensure Priority is numeric and has sensible default
        if 'Priority' not in schedule_with_details.columns:
            schedule_with_details['Priority'] = 3
        schedule_with_details['Priority'] = schedule_with_details['Priority'].fillna(3).astype(int)
        
        # Calculate standard metrics
        self.metrics = calculate_metrics(schedule_with_details, self.operations, 'GA')
        
        # Normalize metrics for fitness calculation (lower is better, scale 0-1)
        makespan_norm = 1.0 / (1.0 + self.metrics.get('Makespan_Days', 100))
        tardiness_norm = 1.0 / (1.0 + self.metrics.get('Total_Tardiness_Days', 100))
        cost_norm = 1.0 / (1.0 + self.metrics.get('Total_Cost_$', 10000) / 1000)
        utilization_score = self.metrics.get('Machine_Utilization_%', 0) / 100.0
        
        # Weighted fitness
        self.fitness_score = (
            weights['makespan'] * makespan_norm +
            weights['tardiness'] * tardiness_norm +
            weights['cost'] * cost_norm +
            weights['utilization'] * utilization_score
        )
        
        # Penalty for constraint violations
        violation_penalty = len(self.constraint_violations) * 0.1
        self.fitness_score -= violation_penalty
        
        # Store breakdown for explainability
        self.fitness_breakdown = {
            'makespan_days': self.metrics.get('Makespan_Days', 0),
            'makespan_contribution': weights['makespan'] * makespan_norm,
            'tardiness_days': self.metrics.get('Total_Tardiness_Days', 0),
            'tardiness_contribution': weights['tardiness'] * tardiness_norm,
            'total_cost': self.metrics.get('Total_Cost_$', 0),
            'cost_contribution': weights['cost'] * cost_norm,
            'utilization_pct': self.metrics.get('Machine_Utilization_%', 0),
            'utilization_contribution': weights['utilization'] * utilization_score,
            'constraint_violations': len(self.constraint_violations),
            'violation_penalty': violation_penalty,
            'final_fitness': self.fitness_score
        }
        
        return self.fitness_score


class GeneticAlgorithmScheduler:
    """
    Main GA Optimizer for Job Shop Scheduling
    
    Evolution Process:
    1. Create initial population (random valid schedules)
    2. Evaluate fitness of each chromosome
    3. Select best chromosomes to be parents
    4. Create offspring through crossover (combining parent genes)
    5. Mutate some genes randomly (exploration)
    6. Replace worst chromosomes with offspring
    7. Repeat for N generations
    
    Result: Near-optimal schedule that balances multiple objectives
    """
    
    def __init__(self, operations: pd.DataFrame, machines: pd.DataFrame,
                 df_effective: pd.DataFrame, df_penalties: pd.DataFrame,
                 population_size=50, generations=100, mutation_rate=0.1,
                 crossover_rate=0.8, cost_threshold=0.9):
        
        self.operations = operations
        self.machines = machines
        self.df_effective = df_effective
        self.df_penalties = df_penalties
        
        # GA Parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.cost_threshold = cost_threshold
        
        # Population
        self.population: List[Chromosome] = []
        self.best_chromosome: Optional[Chromosome] = None
        
        # Evolution tracking for explainability
        self.evolution_history = []
        self.generation_stats = []
        
    def initialize_population(self):
        """
        Create initial population of random but valid schedules
        
        Why random? Diversity helps GA explore the solution space
        """
        self.population = []
        for i in range(self.population_size):
            chromosome = Chromosome(
                self.operations, self.machines,
                self.df_effective, self.df_penalties
            )
            chromosome.initialize_random(self.cost_threshold)
            self.population.append(chromosome)

        # Evaluate initial population in parallel (threads are effective with pandas/numpy)
        max_workers = min(8, max(1, self.population_size // 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(c.calculate_fitness) for c in self.population]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.exception(f"Error during initial fitness evaluation: {e}")
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness_score, reverse=True)
        self.best_chromosome = self.population[0]
    
    def selection(self, tournament_size=5) -> Chromosome:
        """
        Tournament Selection: Pick the best from random subset
        
        Why tournament? Simple, effective, and maintains diversity
        Better individuals have higher chance but not guaranteed
        """
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness_score)
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Uniform Crossover: Mix genes from two parents
        
        How it works:
        - For each operation, randomly pick machine assignment from parent1 or parent2
        - Child inherits good traits from both parents
        
        Example:
          Parent1: J101_Op1 -> M1, J101_Op2 -> M3
          Parent2: J101_Op1 -> M4, J101_Op2 -> M3
          Child:   J101_Op1 -> M4 (from P2), J101_Op2 -> M3 (from P1 or P2)
        """
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        child1 = Chromosome(self.operations, self.machines, self.df_effective, self.df_penalties)
        child2 = Chromosome(self.operations, self.machines, self.df_effective, self.df_penalties)
        
        child1.genes = {}
        child2.genes = {}
        
        for op_id in parent1.genes:
            if random.random() < 0.5:
                child1.genes[op_id] = parent1.genes[op_id]
                child2.genes[op_id] = parent2.genes[op_id]
            else:
                child1.genes[op_id] = parent2.genes[op_id]
                child2.genes[op_id] = parent1.genes[op_id]
        
        return child1, child2
    
    def mutate(self, chromosome: Chromosome):
        """
        Mutation: Randomly change some genes
        
        Types of mutations:
        1. Change machine assignment (within eligible machines)
        2. Swap sequence positions of two operations on same machine
        3. Flip make-or-buy decision
        
        Why mutate? Prevents getting stuck in local optimum
        """
        for op_id, (machine_id, position, outsource) in list(chromosome.genes.items()):
            if random.random() < self.mutation_rate:
                op = self.operations[self.operations['Operation_ID'] == op_id].iloc[0]
                op_type = op['Op_Type']
                eligible = get_eligible_machines(op_type)
                
                mutation_type = random.choice(['machine', 'position', 'outsource'])
                
                if mutation_type == 'machine' and eligible and len(eligible) > 1:
                    # Change machine
                    new_machine = random.choice([m for m in eligible if m != machine_id])
                    chromosome.genes[op_id] = (new_machine, position, outsource)
                    chromosome.gene_explanations[op_id] = f"Mutated: changed from {machine_id} to {new_machine}"
                
                elif mutation_type == 'position':
                    # Change position in queue
                    new_position = random.randint(0, 100)
                    chromosome.genes[op_id] = (machine_id, new_position, outsource)
                    chromosome.gene_explanations[op_id] = f"Mutated: moved from position {position} to {new_position}"
                
                elif mutation_type == 'outsource' and op.get('Outsource_Flag') == 'Y':
                    # Flip outsource decision
                    new_outsource = 'OUTSOURCE' if outsource == 'IN_HOUSE' else 'IN_HOUSE'
                    if new_outsource == 'OUTSOURCE':
                        chromosome.genes[op_id] = ('OUTSOURCE', 0, 'OUTSOURCE')
                    else:
                        new_machine = random.choice(eligible) if eligible else machine_id
                        chromosome.genes[op_id] = (new_machine, position, 'IN_HOUSE')
                    chromosome.gene_explanations[op_id] = f"Mutated: flipped from {outsource} to {new_outsource}"
    
    def evolve(self):
        """
        Main evolution loop
        
        Process:
        1. Evaluate all chromosomes
        2. Select parents
        3. Create offspring (crossover + mutation)
        4. Replace worst with offspring
        5. Track statistics
        6. Repeat
        """
        # Start evolution
        logger.info(f"Initializing population (size={self.population_size})")
        self.initialize_population()

        no_progress_counter = 0
        last_best = self.best_chromosome.fitness_score if self.best_chromosome else float('-inf')

        for generation in range(self.generations):
            try:
                # Track stats for this generation
                fitness_scores = [c.fitness_score for c in self.population]
                avg_fitness = np.mean(fitness_scores)
                best_fitness = max(fitness_scores)
                worst_fitness = min(fitness_scores)

                gen_stats = {
                    'generation': generation,
                    'best_fitness': best_fitness,
                    'avg_fitness': avg_fitness,
                    'worst_fitness': worst_fitness,
                    'best_makespan': self.population[0].metrics.get('Makespan_Days', 0),
                    'best_tardiness': self.population[0].metrics.get('Total_Tardiness_Days', 0),
                    'best_cost': self.population[0].metrics.get('Total_Cost_$', 0),
                    'best_utilization': self.population[0].metrics.get('Machine_Utilization_%', 0)
                }
                self.generation_stats.append(gen_stats)

                logger.info(f"Generation {generation}: best={best_fitness:.6f} avg={avg_fitness:.6f} worst={worst_fitness:.6f}")

                # If no improvement for many generations, allow early exit
                if best_fitness <= last_best:
                    no_progress_counter += 1
                else:
                    no_progress_counter = 0
                    last_best = best_fitness

                if no_progress_counter >= max(10, int(self.generations * 0.1)):
                    logger.info(f"No improvement for {no_progress_counter} generations — stopping early at generation {generation}")
                    break

                # Create next generation
                offspring = []

                inner_guard = 0
                while len(offspring) < self.population_size:
                    inner_guard += 1
                    if inner_guard > self.population_size * 10:
                        logger.warning("Inner offspring loop exceeded guard limit — breaking to avoid infinite loop")
                        break

                    # Selection
                    parent1 = self.selection()
                    parent2 = self.selection()

                    # Crossover
                    try:
                        child1, child2 = self.crossover(parent1, parent2)
                    except Exception as e:
                        logger.exception(f"Error during crossover: {e}")
                        # fallback to parents
                        child1, child2 = parent1, parent2

                    # Mutation
                    try:
                        self.mutate(child1)
                        self.mutate(child2)
                    except Exception as e:
                        logger.exception(f"Error during mutation: {e}")

                    # Defer evaluation: collect children then evaluate in batch for performance
                    offspring.extend([child1, child2])

                # Evaluate offspring fitness in parallel
                if offspring:
                    try:
                        max_workers_children = min(8, max(1, len(offspring) // 4))
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_children) as ex2:
                            futs = [ex2.submit(c.calculate_fitness) for c in offspring]
                            for f in concurrent.futures.as_completed(futs):
                                try:
                                    f.result()
                                except Exception as e:
                                    logger.exception(f"Error during offspring fitness evaluation: {e}")
                    except Exception:
                        logger.exception("Parallel offspring evaluation failed, falling back to sequential")
                        for c in offspring:
                            try:
                                c.calculate_fitness()
                            except Exception:
                                logger.exception("Child fitness calc failed")

                # Elitism: Keep best from previous generation
                self.population.sort(key=lambda x: x.fitness_score, reverse=True)
                elite_count = max(2, self.population_size // 10)
                elite = self.population[:elite_count]

                # Replace population (keep elite + best offspring)
                offspring.sort(key=lambda x: x.fitness_score, reverse=True)
                self.population = elite + offspring[:self.population_size - elite_count]

                # Update best
                self.best_chromosome = self.population[0]

            except Exception as e:
                logger.exception(f"Unhandled exception in generation loop: {e}")
                break

        logger.info(f"Evolution complete. Best fitness={self.best_chromosome.fitness_score:.6f}")
        return self.best_chromosome


def run_ga_optimization(df_ops, df_machines, df_effective, df_penalties,
                       population_size=50, generations=100, mutation_rate=0.1,
                       crossover_rate=0.8, cost_threshold=0.9):
    """
    Wrapper function to run GA optimization
    
    Returns:
    - best_schedule: DataFrame with optimized schedule
    - metrics: Performance metrics
    - evolution_history: Generation-by-generation stats
    - explainability: Why the solution is good
    """
    ga = GeneticAlgorithmScheduler(
        df_ops, df_machines, df_effective, df_penalties,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        cost_threshold=cost_threshold
    )
    logger.info(f"Starting GA optimization: pop={population_size}, gens={generations}, mut={mutation_rate}, cross={crossover_rate}")
    start_time = time.time()
    best_chromosome = None
    try:
        best_chromosome = ga.evolve()
    except Exception as e:
        logger.exception(f"GA evolution failed: {e}")
        # Attempt to recover best available chromosome
        if ga.population:
            best_chromosome = max(ga.population, key=lambda c: c.fitness_score)
        else:
            raise
    end_time = time.time()
    logger.info(f"GA run finished in {end_time - start_time:.1f}s")
    
    # Prepare explainability data
    explainability = {
        'total_generations': generations,
        'final_population_size': len(ga.population),
        'best_fitness': best_chromosome.fitness_score,
        'fitness_breakdown': best_chromosome.fitness_breakdown,
        'constraint_violations': best_chromosome.constraint_violations,
        'gene_sample_explanations': dict(list(best_chromosome.gene_explanations.items())[:10]),
        'evolution_summary': {
            'improvement': ga.generation_stats[-1]['best_fitness'] - ga.generation_stats[0]['best_fitness'],
            'generations_to_convergence': len(ga.generation_stats),
            'final_metrics': best_chromosome.metrics
        }
    }
    
    return {
        'schedule': best_chromosome.schedule_df,
        'metrics': best_chromosome.metrics,
        'evolution_history': ga.generation_stats,
        'explainability': explainability,
        'best_chromosome': best_chromosome
    }
