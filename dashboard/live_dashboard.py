import os, time, httpx
from datetime import datetime
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich import box

API_URL = os.environ.get("API_URL", "http://localhost:8000")
STORE_ID = os.environ.get("STORE_ID", "ST1008")
HOURS = "99999"
REFRESH = 5

console = Console()

def get(path):
    try:
        url = API_URL + path
        if "/metrics" in path or "/funnel" in path or "/heatmap" in path or "/anomalies" in path:
            url += ("&" if "?" in path else "?") + "hours=" + HOURS
        r = httpx.get(url, timeout=4.0)
        if r.status_code < 300:
            return r.json()
    except:
        pass
    return None

def metrics_panel(d):
    if not d:
        return Panel("[red]unavailable[/red]", title="Metrics")
    t = Table.grid(expand=True, padding=(0,2))
    t.add_column(); t.add_column(justify="right")
    cr = d.get("conversion_rate", 0)
    t.add_row("Unique Visitors", "[cyan]" + str(d.get("unique_visitors",0)) + "[/cyan]")
    t.add_row("Conversion Rate", ("[green]" if cr>0.1 else "[yellow]") + str(round(cr*100,1)) + "%[/]")
    t.add_row("Queue Depth", str(d.get("current_queue_depth",0)))
    t.add_row("Abandonment", str(round(d.get("abandonment_rate",0)*100,1)) + "%")
    t.add_row("As Of", "[dim]" + str(d.get("as_of",""))[:19] + "[/dim]")
    return Panel(t, title="Metrics — " + STORE_ID, border_style="blue")

def zones_panel(d):
    if not d or not d.get("avg_dwell_ms_per_zone"):
        return Panel("[dim]No zone data[/dim]", title="Zone Dwell")
    t = Table(box=box.SIMPLE, expand=True)
    t.add_column("Zone", style="cyan"); t.add_column("Dwell", justify="right"); t.add_column("Visits", justify="right")
    for z in sorted(d["avg_dwell_ms_per_zone"], key=lambda x: x["avg_dwell_ms"], reverse=True)[:7]:
        t.add_row(z["zone_id"], str(round(z["avg_dwell_ms"]/1000,1))+"s", str(z["visit_count"]))
    return Panel(t, title="Zone Dwell", border_style="magenta")

def funnel_panel(d):
    if not d:
        return Panel("[red]unavailable[/red]", title="Funnel")
    t = Table(box=box.SIMPLE, expand=True)
    t.add_column("Stage", style="yellow"); t.add_column("Count", justify="right"); t.add_column("Drop-off", justify="right")
    for s in d.get("stages", []):
        dp = s["drop_off_pct"]
        c = "green" if dp<20 else "yellow" if dp<50 else "red"
        t.add_row(s["stage"], str(s["count"]), ("["+c+"]"+str(dp)+"%[/]") if dp>0 else "-")
    return Panel(t, title="Funnel", border_style="yellow")

def anomalies_panel(d):
    if not d:
        return Panel("[red]unavailable[/red]", title="Anomalies")
    items = d.get("anomalies", [])
    if not items:
        return Panel(Align.center("[green]No anomalies[/green]"), title="Anomalies", border_style="green")
    t = Table(box=box.SIMPLE, expand=True)
    t.add_column("Sev", width=8); t.add_column("Type"); t.add_column("Description")
    colors = {"INFO":"dim","WARN":"yellow","CRITICAL":"bold red"}
    for a in items:
        c = colors.get(a["severity"],"white")
        t.add_row("["+c+"]"+a["severity"]+"[/]", a["type"], a["description"][:55]+"...")
    return Panel(t, title="Anomalies ("+str(len(items))+")", border_style="red")

def health_panel(d):
    if not d:
        return Panel("[red]API unreachable[/red]", title="Health")
    s = d.get("status","?")
    c = "green" if s=="ok" else "yellow" if s=="degraded" else "red"
    lines = ["["+c+"]"+s.upper()+"[/]  DB: "+d.get("database","?")]
    for st in d.get("stores",[]):
        sc = "green" if st["status"]=="OK" else "red"
        lines.append("  ["+sc+"]"+st["store_id"]+": "+st["status"]+"[/]")
    return Panel("\n".join(lines), title="Health", border_style=c)

spin = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
tick = 0
with Live(console=console, refresh_per_second=1, screen=True) as live:
    while True:
        m = get("/stores/"+STORE_ID+"/metrics")
        f = get("/stores/"+STORE_ID+"/funnel")
        a = get("/stores/"+STORE_ID+"/anomalies")
        h = get("/health")
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=1),
        )
        layout["body"].split_row(Layout(name="left"), Layout(name="right"))
        layout["left"].split_column(Layout(name="metrics"), Layout(name="zones"))
        layout["right"].split_column(Layout(name="funnel"), Layout(name="anomalies"), Layout(name="health", size=6))
        layout["header"].update(Panel(Align.center("[bold]Store Intelligence Dashboard[/bold]  [dim]" + spin[tick%10] + " refreshing every "+str(REFRESH)+"s  "+datetime.now().strftime("%H:%M:%S")+"[/dim]"), border_style="bright_blue"))
        layout["metrics"].update(metrics_panel(m))
        layout["zones"].update(zones_panel(m))
        layout["funnel"].update(funnel_panel(f))
        layout["anomalies"].update(anomalies_panel(a))
        layout["health"].update(health_panel(h))
        layout["footer"].update(Align.center("[dim]q to quit | store: "+STORE_ID+"[/dim]"))
        live.update(layout)
        tick += 1
        time.sleep(REFRESH)
