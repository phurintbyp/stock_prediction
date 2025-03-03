import numpy as np

# GOOGL data
fcf = [42843, 67012, 60010,	69495, 72764]
wacc = 0.0943 # Can be calculated further
growth_rate = 0.0646  # Can get from other code
debt = 28140
cash = 95660
shares_outstanding = 12210

class DCF():
    
    def __init__(self, fcf, wacc, growth_rate, debt, cash, shares_outstanding):
        self.fcf = fcf
        self.wacc = wacc
        self.growth_rate = growth_rate
        self.debt = debt
        self.cash = cash
        self.shares_outstanding = shares_outstanding
        self.intrinsic_value = []

    def calculate_intri(self):

        for i in range(len(self.growth_rate)):
            years = np.arange(1, len(self.fcf) + 1)
            fcf_n = self.fcf[-1] # FCF value of latest year

            # pv_fcf = sum([fcf[i] / (1 + wacc) ** years[i] for i in range(len(fcf))])
            self.pv_fcf = [self.fcf[j] / (1 + self.wacc) ** years[j] for j in range(len(self.fcf))]
            self.total_pv_fcf = sum(self.pv_fcf)

            self.terminal_value = (fcf_n * (1 + self.growth_rate[i])) / (self.wacc - self.growth_rate[i])

            self.pv_terminal_value = self.terminal_value / (1 + self.wacc) ** len(self.fcf)

            self.enterprise_value = self.total_pv_fcf + self.pv_terminal_value

            self.equity_value = self.enterprise_value - self.debt

            self.intrinsic_value.append(self.equity_value / self.shares_outstanding)

        return self.intrinsic_value

if __name__ == "__main__":

    dcf = DCF(fcf, wacc, growth_rate, debt, cash, shares_outstanding)
    intrinsic_value = dcf.calculate_intri()
    print(f"Discounted Free Cash Flows (PV_FCF): {[round(val, 2) for val in dcf.pv_fcf]}")
    print(f"Total Present Value of FCF: {dcf.total_pv_fcf:.2f}")
    print(f"Terminal Value : {dcf.terminal_value:.2f}")
    print(f"Present value of Terminal value : {dcf.pv_terminal_value:.2f}")
    print(f"Enterprise value : {dcf.enterprise_value:.2f}")
    print(f"Equity Value: ${dcf.equity_value:.2f}M")
    print(f"Intrinsic Value per Share: ${intrinsic_value:.2f}")