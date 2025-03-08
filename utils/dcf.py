import numpy as np
import matplotlib.pyplot as plt

class DCF():
    
    def __init__(self, fcf, wacc, growth_rate, debt, cash, shares_outstanding):
        self.fcf = fcf
        self.wacc = wacc
        self.growth_rate = growth_rate
        self.debt = debt
        self.cash = cash
        self.shares_outstanding = shares_outstanding
        self.intrinsic_value = []
        self.year = []

    def calculate_intri(self):
        terminal_growth = 0.02  # Terminal growth rate (2% is typical)
        forecast_period = 5  # First stage forecast period
        
        for i in range(len(self.growth_rate)):
            current_fcf = self.fcf[-1]
            projected_fcf = []
            
            # Stage 1: Growth phase
            for year in range(forecast_period):
                current_fcf *= (1 + self.growth_rate[i])
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
            share_value = equity_value / self.shares_outstanding
            
            self.intrinsic_value.append(float(share_value))
            self.year.append(int(i))

    def plot_data(self):
        print(self.intrinsic_value)
        plt.figure(figsize=(12, 6))
        plt.plot(self.year, self.intrinsic_value, marker="o", linestyle="-", linewidth=2, color="red", label="DCF Intrinsic Value")
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.xticks(np.array(self.year, dtype=int)[::2])
        plt.title("DCF Intrinsic Value Data")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

    def run(self):
        self.calculate_intri()
        self.plot_data()

if __name__ == "__main__":

    # GOOGL data
    fcf = [42843, 67012, 60010,	69495, 72764]
    wacc = 0.0943 # Can be calculated further
    growth_rate = [0.0646, 0.065, 0.066, 0.067, 0.068, 0.069, 0.07, 0.071, 0.072, 0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.08, 0.081, 0.082, 0.083]  # Example growth rates for testing
    debt = 28140
    cash = 95660
    shares_outstanding = 12210

    dcf = DCF(fcf, wacc, growth_rate, debt, cash, shares_outstanding)
    dcf.run()
    