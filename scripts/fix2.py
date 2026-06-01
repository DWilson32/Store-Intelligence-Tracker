with open("/app/app/metrics.py", "r") as f:
    content = f.read()

if "thirty_min_ago" not in content:
    old = 'window_start = (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")'
    new = 'window_start = (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")\n    thirty_min_ago = (now - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")'
    content = content.replace(old, new)
    with open("/app/app/metrics.py", "w") as f:
        f.write(content)
    print("Fixed")
else:
    print("Already present")
