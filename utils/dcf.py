import numpy as np
import matplotlib.pyplot as plt

class DCF():
    def __init__(self, fcf, wacc, growth_rate, debt, cash, shares_outstanding):
        self.fcf = fcf
        self.wacc = wacc
        self.growth_rate = growth_rate  # Now expecting a single value
        self.debt = debt
        self.cash = cash
        self.shares_outstanding = shares_outstanding
        self.intrinsic_value = None

    def calculate_intri(self):
        terminal_growth = 0.02  # Terminal growth rate (2% is typical)
        forecast_period = 5  # First stage forecast period
        
        current_fcf = self.fcf[-1]
        projected_fcf = []
        
        # Stage 1: Growth phase
        for year in range(forecast_period):
            current_fcf *= (1 + self.growth_rate)
            projected_fcf.append(current_fcf)
        
        # Calculate PV of explicit forecast period
        pv_forecast = sum([
            fcf / ((1 + self.wacc) ** (t + 1))
            for t, fcf in enumerate(projected_fcf)
        ])
        
        # Calculate terminal value using Gordon Growth model
        terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (self.wacc - terminal_growth)
        
        # Discount terminal value to present
        pv_terminal = terminal_value / ((1 + self.wacc) ** forecast_period)
        
        # Calculate enterprise value
        enterprise_value = pv_forecast + pv_terminal
        
        # Calculate equity value
        equity_value = enterprise_value - self.debt + self.cash
        
        # Calculate per share value
        self.intrinsic_value = float(equity_value / self.shares_outstanding)  # Convert to float

    def plot_data(self):
        print(f"Calculated Intrinsic Value: ${float(self.intrinsic_value):.2f}")  # Convert to float before formatting

    def run(self):
        self.calculate_intri()
        self.plot_data()

if __name__ == "__main__":

    # GOOGL data
    fcf = [42843, 67012, 60010, 69495, 72764]
    wacc = 0.0943 # Can be calculated further
    growth_rate = 0.07  # Example growth rate for testing
    debt = 28140
    cash = 95660
    shares_outstanding = 12210

    dcf = DCF(fcf, wacc, growth_rate, debt, cash, shares_outstanding)
    dcf.run()
