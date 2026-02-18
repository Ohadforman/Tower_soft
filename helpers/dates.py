from datetime import datetime, timedelta
def compute_next_planned_draw_date(now_dt: datetime = None) -> str:
    now_dt = now_dt or datetime.now()
    wd = now_dt.weekday()  # Mon=0 ... Sun=6
    if wd == 3:  # Thursday
        next_dt = now_dt + timedelta(days=3)  # Sunday
    else:
        next_dt = now_dt + timedelta(days=1)
    return next_dt.strftime("%Y-%m-%d")