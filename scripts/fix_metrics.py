with open("/app/app/metrics.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
inserted = False
for line in lines:
    if "# 1. Unique visitors" in line and not inserted:
        skip = True
        new_lines.append("    # 1. Unique visitors deduplicated with 30s window\n")
        new_lines.append("    entry_rows = (await session.execute(\n")
        new_lines.append("        select(EventRecord.visitor_id, EventRecord.timestamp).where(\n")
        new_lines.append("            EventRecord.store_id == store_id,\n")
        new_lines.append("            EventRecord.event_type.in_(['ENTRY', 'REENTRY']),\n")
        new_lines.append("            EventRecord.is_staff == False,\n")
        new_lines.append("            EventRecord.timestamp >= window_start,\n")
        new_lines.append("        ).order_by(EventRecord.timestamp)\n")
        new_lines.append("    )).fetchall()\n")
        new_lines.append("    unique_visitors = 0\n")
        new_lines.append("    last_entry_dt = None\n")
        new_lines.append("    for _, ts_str in entry_rows:\n")
        new_lines.append("        dt = _parse_ts(ts_str)\n")
        new_lines.append("        if dt is None:\n")
        new_lines.append("            continue\n")
        new_lines.append("        if last_entry_dt is None or (dt - last_entry_dt).total_seconds() > 30:\n")
        new_lines.append("            unique_visitors += 1\n")
        new_lines.append("            last_entry_dt = dt\n")
        inserted = True
        continue
    if skip:
        if "scalar() or 0" in line:
            skip = False
        continue
    new_lines.append(line)

with open("/app/app/metrics.py", "w") as f:
    f.writelines(new_lines)
print("Done, inserted:", inserted)
