import numpy as np

fcf = [2126, 14895, 40319, 57180, 88597]
years = np.arange(1, len(fcf) + 1)
wacc = 0.078 # Can be calculated further
growth_rate = 0.03  # Can get from other code
fcf_n = fcf[-1] # FCF value of latest year
debt = 47556
cash = 66385
shares_outstanding = 10456

# pv_fcf = sum([fcf[i] / (1 + wacc) ** years[i] for i in range(len(fcf))])
pv_fcf = [fcf[i] / (1 + wacc) ** years[i] for i in range(len(fcf))]
total_pv_fcf = sum(pv_fcf)

terminal_value = (fcf_n * (1 + growth_rate)) / (wacc - growth_rate)

pv_terminal_value = terminal_value / (1 + wacc) ** len(fcf)

enterprise_value = total_pv_fcf + pv_terminal_value

equity_value = enterprise_value - debt

intrinsic_value = equity_value / shares_outstanding

print(f"Discounted Free Cash Flows (PV_FCF): {[round(val, 2) for val in pv_fcf]}")
print(f"Total Present Value of FCF: {total_pv_fcf:.2f}")
print(f"Terminal Value : {terminal_value:.2f}")
print(f"Present value of Terminal value : {pv_terminal_value:.2f}")
print(f"Enterprise value : {enterprise_value:.2f}")
print(f"Equity Value: ${equity_value:.2f}M")
print(f"Intrinsic Value per Share: ${intrinsic_value:.2f}")