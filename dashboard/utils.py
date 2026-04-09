def format_currency(val, decimals=2):
    try:
        return f"₹{val:,.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def signal_color(signal: str) -> str:
    if signal == "BUY":
        return "#00e676"
    if signal == "SELL":
        return "#ff5252"
    return "#cccccc"
