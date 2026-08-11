def simulate_scalp(buy_price: float, qty: int, target_percent: float = 0.005):
    # Calculate the dynamic target tick based on the 0.5% move
    target_tick = buy_price * target_percent
    sell_price = buy_price + target_tick
    
    buy_turnover = buy_price * qty
    sell_turnover = sell_price * qty
    total_turnover = buy_turnover + sell_turnover
    
    # 1. Brokerage: 0.03% or Rs. 20 whichever is lower (per leg for Intraday)
    buy_brokerage = min(20.0, buy_turnover * 0.0003)
    sell_brokerage = min(20.0, sell_turnover * 0.0003)
    total_brokerage = buy_brokerage + sell_brokerage
    
    # 2. STT: 0.025% on the SELL side only for Intraday Equity
    stt = sell_turnover * 0.00025
    
    # 3. Exchange Transaction Charges (NSE): ~0.00322% on total turnover
    etc = total_turnover * 0.0000322
    
    # 4. GST: 18% on (Brokerage + ETC)
    gst = (total_brokerage + etc) * 0.18
    
    # 5. SEBI Charges: Rs. 10 per crore (0.0001%) on total turnover
    sebi = total_turnover * 0.000001
    
    # 6. Stamp Duty: 0.003% on the BUY side only
    stamp_duty = buy_turnover * 0.00003
    
    # Calculate Totals
    total_fees = total_brokerage + stt + etc + gst + sebi + stamp_duty
    gross_profit = (sell_price - buy_price) * qty
    net_profit = gross_profit - total_fees
    
    print(f"--- Trade Simulation (0.5% Target) ---")
    print(f"Buy: {qty} shares @ {buy_price} INR")
    print(f"Sell: {qty} shares @ {sell_price:.2f} INR (Target Move: +{target_tick:.2f} INR)")
    print(f"Gross Profit:  {gross_profit:.2f} INR")
    print(f"Total Fees:    {total_fees:.2f} INR")
    print(f"Net Profit:    {net_profit:.2f} INR")
    
    if net_profit > 0:
        print("Result: PROFITABLE ✅")
    else:
        print("Result: LOSS ❌ (Fees exceed 0.5% move)")

if __name__ == "__main__":
    # Simulating a scalp on a stock priced at 105.5 with 100 shares (Your original example!)
    simulate_scalp(buy_price=105.5, qty=100)
    
    # Simulating a scalp on a stock priced at 500 with 100 shares
    print("\n")
    simulate_scalp(buy_price=500.0, qty=100)