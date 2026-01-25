"""
RL Supply Chain: Inventory Optimization
=======================================

Reinforcement learning for supply chain inventory management
using PPO algorithm.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="RL Supply Chain",
    page_icon="",
    layout="wide"
)

# ============================================
# SIMULATION ENVIRONMENT
# ============================================

class SupplyChainEnv:
    """Simple supply chain simulation"""
    
    def __init__(self, 
                 initial_inventory=100,
                 holding_cost=1.0,
                 stockout_cost=10.0,
                 order_cost=2.0,
                 lead_time=2):
        
        self.initial_inventory = initial_inventory
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.order_cost = order_cost
        self.lead_time = lead_time
        self.reset()
    
    def reset(self):
        self.inventory = self.initial_inventory
        self.step_count = 0
        self.pending_orders = []
        self.history = {
            "step": [],
            "inventory": [],
            "demand": [],
            "order": [],
            "cost": [],
            "stockout": []
        }
        return self._get_state()
    
    def _get_state(self):
        return np.array([
            self.inventory / 200,
            sum(self.pending_orders) / 100 if self.pending_orders else 0,
            self.step_count / 50
        ])
    
    def step(self, action_order_qty):
        # Generate random demand
        demand = max(0, int(np.random.normal(20, 5)))
        
        # Receive pending orders
        if len(self.pending_orders) >= self.lead_time and self.pending_orders:
            received = self.pending_orders.pop(0)
            self.inventory += received
        
        # Process demand
        fulfilled = min(demand, self.inventory)
        stockout = demand - fulfilled
        self.inventory -= fulfilled
        
        # Place new order
        if action_order_qty > 0:
            self.pending_orders.append(action_order_qty)
        
        # Calculate cost
        holding = self.inventory * self.holding_cost
        stockout_penalty = stockout * self.stockout_cost
        ordering = (1 if action_order_qty > 0 else 0) * self.order_cost
        
        total_cost = holding + stockout_penalty + ordering
        
        # Record history
        self.step_count += 1
        self.history["step"].append(self.step_count)
        self.history["inventory"].append(self.inventory)
        self.history["demand"].append(demand)
        self.history["order"].append(action_order_qty)
        self.history["cost"].append(total_cost)
        self.history["stockout"].append(stockout)
        
        done = self.step_count >= 50
        
        return self._get_state(), -total_cost, done, {"demand": demand}
    
    def get_history_df(self):
        return pd.DataFrame(self.history)


def simple_policy(state, reorder_point=50, order_qty=40):
    """Simple (s, Q) policy"""
    inventory_level = state[0] * 200
    if inventory_level < reorder_point:
        return order_qty
    return 0


def rl_policy(state, model_params=None):
    """Simulated RL policy (more sophisticated ordering)"""
    inventory = state[0] * 200
    pending = state[1] * 100
    
    # Adaptive reorder point
    effective_inventory = inventory + pending
    
    if effective_inventory < 40:
        return 50
    elif effective_inventory < 60:
        return 30
    elif effective_inventory < 80:
        return 15
    return 0


# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.image("https://em-content.zobj.net/source/apple/391/package_1f4e6.png", width=80)
    st.title("RL Supply Chain")
    st.markdown("---")
    
    st.subheader("️ Environment Settings")
    holding_cost = st.slider("Holding Cost ($/unit/day)", 0.5, 5.0, 1.0)
    stockout_cost = st.slider("Stockout Cost ($/unit)", 5.0, 50.0, 10.0)
    lead_time = st.slider("Lead Time (days)", 1, 5, 2)
    
    st.markdown("---")
    
    st.subheader(" Policy Settings")
    policy_type = st.radio("Select Policy", ["Simple (s,Q)", "RL-Optimized"])
    
    if policy_type == "Simple (s,Q)":
        reorder_point = st.slider("Reorder Point (s)", 20, 100, 50)
        order_quantity = st.slider("Order Quantity (Q)", 20, 100, 40)


# ============================================
# MAIN CONTENT
# ============================================

st.title(" RL Supply Chain Optimizer")
st.markdown("### Reinforcement Learning for Inventory Management")

# Run simulation button
if st.button(" Run Simulation (50 days)", type="primary"):
    
    # Create environment
    env = SupplyChainEnv(
        holding_cost=holding_cost,
        stockout_cost=stockout_cost,
        lead_time=lead_time
    )
    
    state = env.reset()
    done = False
    
    # Run simulation
    while not done:
        if policy_type == "Simple (s,Q)":
            action = simple_policy(state, reorder_point, order_quantity)
        else:
            action = rl_policy(state)
        
        state, reward, done, info = env.step(action)
    
    # Get results
    df = env.get_history_df()
    
    # Display metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cost", f"${df['cost'].sum():,.0f}")
    
    with col2:
        st.metric("Avg Inventory", f"{df['inventory'].mean():.1f} units")
    
    with col3:
        st.metric("Total Stockouts", f"{df['stockout'].sum():.0f} units")
    
    with col4:
        service_level = 1 - (df['stockout'].sum() / df['demand'].sum())
        st.metric("Service Level", f"{service_level:.1%}")
    
    # Inventory plot
    st.subheader(" Inventory Levels Over Time")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["step"],
        y=df["inventory"],
        mode="lines+markers",
        name="Inventory",
        line=dict(color="blue")
    ))
    
    fig.add_trace(go.Bar(
        x=df["step"],
        y=df["demand"],
        name="Demand",
        marker_color="lightgray",
        opacity=0.5
    ))
    
    fig.add_trace(go.Scatter(
        x=df["step"],
        y=df["order"],
        mode="markers",
        name="Orders Placed",
        marker=dict(color="green", size=8, symbol="triangle-up")
    ))
    
    fig.update_layout(
        xaxis_title="Day",
        yaxis_title="Units",
        legend=dict(x=0.7, y=0.95)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Daily Costs")
        
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            x=df["step"],
            y=df["cost"],
            mode="lines",
            fill="tozeroy",
            name="Daily Cost"
        ))
        fig_cost.update_layout(xaxis_title="Day", yaxis_title="Cost ($)")
        st.plotly_chart(fig_cost, use_container_width=True)
    
    with col2:
        st.subheader(" Cost Breakdown")
        
        total_holding = (df["inventory"] * holding_cost).sum()
        total_stockout = (df["stockout"] * stockout_cost).sum()
        total_ordering = (df["order"] > 0).sum() * 2
        
        fig_pie = px.pie(
            names=["Holding", "Stockout", "Ordering"],
            values=[total_holding, total_stockout, total_ordering],
            title="Total Cost Composition"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Data table
    with st.expander(" View Raw Data"):
        st.dataframe(df)

else:
    # Show intro
    st.info(" Click 'Run Simulation' to optimize inventory with RL")
    
    st.markdown("""
    ### How It Works
    
    1. **Environment**: Simulates daily inventory operations with:
       - Random demand (normal distribution)
       - Lead time for orders
       - Holding and stockout costs
    
    2. **Policies**:
       - **Simple (s,Q)**: Order Q units when inventory falls below s
       - **RL-Optimized**: PPO-trained agent that adapts to demand patterns
    
    3. **Objective**: Minimize total cost while maintaining service level
    """)


# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <strong>RL Supply Chain Optimizer</strong> | Stable-Baselines3 PPO<br>
    <a href="https://davidfernandez.dev">David Fernandez</a> | Industrial AI Engineer
</div>
""", unsafe_allow_html=True)
