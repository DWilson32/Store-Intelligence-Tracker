import httpx
import json
import os

EVENT_DIR = "/data/events"
API_URL = "http://localhost:8000"

all_events = []

for fname in os.listdir(EVENT_DIR):
    if fname.endswith(".jsonl"):
        path = os.path.join(EVENT_DIR, fname)
        with open(path) as f:
            events = [json.loads(line) for line in f if line.strip()]
        print(fname + ": " + str(len(events)) + " events")
        all_events.extend(events)

print("Total events: " + str(len(all_events)))

accepted = 0
for i in range(0, len(all_events), 500):
    batch = all_events[i:i+500]
    r = httpx.post(API_URL + "/events/ingest", json={"events": batch})
    data = r.json()
    accepted += data.get("accepted", 0)
    print("Batch " + str(i//500+1) + ": accepted=" + str(data.get("accepted")))

print("Done! Total accepted: " + str(accepted))
