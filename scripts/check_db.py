import sqlite3
c = sqlite3.connect('data/store_intelligence.db').cursor()

c.execute("SELECT COUNT(DISTINCT visitor_id) FROM events WHERE event_type IN ('ENTRY','REENTRY') AND camera_id LIKE '%ENTRY%'")
print('ENTRY camera visitors:', c.fetchone()[0])

c.execute("SELECT DISTINCT camera_id FROM events WHERE event_type='ENTRY'")
print('Cameras with ENTRY events:', [r[0] for r in c.fetchall()])

c.execute("SELECT camera_id, COUNT(*) FROM events WHERE event_type='ENTRY' GROUP BY camera_id")
print('ENTRY events per camera:')
for r in c.fetchall():
    print(f'  {r[0]}: {r[1]}')
