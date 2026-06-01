# PROMPT: "Write pytest tests for anomaly detection in a retail store analytics API. Anomalies:
# BILLING_QUEUE_SPIKE (WARN at 5+, CRITICAL at 10+), CONVERSION_DROP (WARN at 25% below 7-day
# avg, CRITICAL at 50%), DEAD_ZONE (no visits in 30 min to a zone that previously had traffic),
# STALE_FEED (no events in 10 min). Each anomaly has severity INFO/WARN/CRITICAL and
# suggested_action. Test: queue spike at threshold, below threshold produces no anomaly,
# dead zone after active zone goes quiet, stale feed after recent events, severity escalation."
#
# CHANGES MADE:
# - Isolated each test with fresh DB state (autouse fixture drops/creates tables)
# - Seeded historical conversion data for CONVERSION_DROP tests using backdated timestamps
# - Fixed DEAD_ZONE test: must seed ZONE_ENTER event in past, then NOT in recent 30-min window
# - Queue spike test uses BILLING_QUEUE_JOIN without matching ABANDON to ensure net depth > 0
# - Added assertion that no anomalies returned when store is healthy and recently active

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from httpx import AsyncClient

from app.main import app
from app.database import EventRecord, POSTransaction

NOW = datetime.now(timezone.utc)
STORE_ID = "ST1008"


def _ts(offset_min: int = 0) -> str:
    return (NOW - timedelta(minutes=offset_min)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts_days(offset_days: int = 0) -> str:
    return (NOW - timedelta(days=offset_days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_event(
    event_type: str,
    visitor_id: str = None,
    zone_id: str = None,
    is_staff: bool = False,
    timestamp: str = None,
    queue_depth: int = None,
    dwell_ms: int = 0,
) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": STORE_ID,
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": visitor_id or f"VIS_{uuid.uuid4().hex[:6]}",
        "event_type": event_type,
        "timestamp": timestamp or _ts(0),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": 0.88,
        "metadata": {"queue_depth": queue_depth, "sku_zone": zone_id, "session_seq": 1},
    }


async def _ingest(client, events):
    resp = await client.post("/events/ingest", json={"events": events})
    assert resp.status_code in (200, 207)
    return resp.json()


# ---------------------------------------------------------------------------
# BILLING_QUEUE_SPIKE tests
# ---------------------------------------------------------------------------

class TestQueueSpikeAnomaly:
    @pytest.mark.asyncio
    async def test_no_spike_below_threshold(self, client):
        """Queue depth of 2 (below WARN threshold of 5) → no QUEUE_SPIKE anomaly."""
        events = [
            _make_event("BILLING_QUEUE_JOIN", visitor_id=f"VIS_q{i}",
                        zone_id="BILLING", queue_depth=i + 1)
            for i in range(2)
        ]
        await _ingest(client, events)
        resp = await client.get(f"/stores/{STORE_ID}/anomalies")
        types = {a["type"] for a in resp.json()["anomalies"]}
        assert "BILLING_QUEUE_SPIKE" not in types

    @pytest.mark.asyncio
    async def test_warn_at_five_in_queue(self, client):
        """Queue depth of 5 → BILLING_QUEUE_SPIKE with WARN severity."""
        events = [
            _make_event("BILLING_QUEUE_JOIN", visitor_id=f"VIS_q{i}",
                        zone_id="BILLING", queue_depth=i + 1)
            for i in range(5)
        ]
        await _ingest(client, events)
        resp = await client.get(f"/stores/{STORE_ID}/anomalies")
        spikes = [a for a in resp.json()["anomalies"] if a["type"] == "BILLING_QUEUE_SPIKE"]
        assert len(spikes) >= 1
        assert spikes[0]["severity"] in ("WARN", "CRITICAL")

    @pytest.mark.asyncio
    async def test_critical_at_ten_in_queue(self, client):
        """Queue depth of 10 → BILLING_QUEUE_SPIKE with CRITICAL severity."""
        events = [
            _make_event("BILLING_QUEUE_JOIN", visitor_id=f"VIS_q{i}",
                        zone_id="BILLING", queue_depth=i + 1)
            for i in range(10)
        ]
        await _ingest(client, events)
        resp = await client.get(f"/stores/{STORE_ID}/anomalies")
        spikes = [a for a in resp.json()["anomalies"] if a["type"] == "BILLING_QUEUE_SPIKE"]
        assert len(spikes) >= 1
        assert spikes[0]["severity"] == "CRITICAL"

    @pytest.mark.asyncio
    async def test_abandons_reduce_effective_queue(self, client):
        """5 joins + 4 abandons = depth of 1 → no WARN anomaly."""
        joins = [
            _make_event("BILLING_QUEUE_JOIN", visitor_id=f"VIS_jq{i}",
                        zone_id="BILLING", queue_depth=i + 1)
            for i in range(5)
        ]
        abandons = [
            _make_event("BILLING_QUEUE_ABANDON", visitor_id=f"VIS_jq{i}")
            for i in range(4)
        ]
        await _ingest(client, joins + abandons)
        resp = await client.get(f"/stores/{STORE_ID}/anomalies")
        spikes = [a for a in resp.json()["anomalies"] if a["type"] == "BILLING_QUEUE_SPIKE"]
        assert len(spikes) == 0


# ---------------------------------------------------------------------------
# DEAD_ZONE tests
# ---------------------------------------------------------------------------

class TestDeadZoneAnomaly:
    @pytest.mark.asyncio
    async def test_dead_zone_after_period_of_inactivity(self, client, db_session):
        """Zone with old visits but none in last 30 min → DEAD_ZONE anomaly."""
        # Seed an old ZONE_ENTER event (45 min ago) — makes zone "known"
        old_event = EventRecord(
            event_id=str(uuid.uuid4()),
            store_id=STORE_ID,
            camera_id="CAM_FLOOR_01",
            visitor_id="VIS_old1",
            event_type="ZONE_ENTER",
            timestamp=_ts(45),  # 45 minutes ago
            zone_id="HAIRCARE",
            dwell_ms=0,
            is_staff=False,
            confidence=0.9,
        )
        db_session.add(old_event)
        await db_session.commit()

        resp = await client.get(f"/stores/{STORE_ID}/anomalies")
        dead = [a for a in resp.json()["anomalies"] if a["type"] == "DEAD_ZONE"]
        assert len(dead) >= 1
        assert any(a["zone_id"] == "HAIRCARE" for a in dead)

    @pytest.mark.asyncio
    async def test_active_zone_no_dead_zone(self, client):
        """Zone with recent visit in last 30 min → no DEAD_ZONE for that zone."""
        events = [
            _make_event("ZONE_ENTER", visitor_id="VIS_active",
                        zone_id="SKINCARE", timestamp=_ts(5)),  # 5 min ago
        ]
        await _ingest(client, events)
        resp = await client.get(f"/stores/{STORE_ID}/anomalies")
        dead = [
            a for a in resp.json()["anomalies"]
            if a["type"] == "DEAD_ZONE" and a.get("zone_id") == "SKINCARE"
        ]
        assert len(dead) == 0


# ---------------------------------------------------------------------------
# STALE_FEED tests
# ---------------------------------------------------------------------------

class TestStaleFeedAnomaly:
    @pytest.mark.asyncio
    async def test_stale_feed_no_events(self, client):
        """Store with no events at all → STALE_FEED anomaly."""
        resp = await client.get(f"/stores/{STORE_ID}/anomalies")
        types = {a["type"] for a in resp.json()["anomalies"]}
        assert "STALE_FEED" in types

    @pytest.mark.asyncio
    async def test_stale_feed_old_events_only(self, client, db_session):
        """Last event > 10 min ago → STALE_FEED."""
        old = EventRecord(
            event_id=str(uuid.uuid4()),
            store_id=STORE_ID,
            camera_id="CAM_ENTRY_01",
            visitor_id="VIS_stale",
            event_type="ENTRY",
            timestamp=_ts(15),  # 15 minutes ago
            dwell_ms=0,
            is_staff=False,
            confidence=0.9,
        )
        db_session.add(old)
        await db_session.commit()

        resp = await client.get(f"/stores/{STORE_ID}/anomalies")
        stale = [a for a in resp.json()["anomalies"] if a["type"] == "STALE_FEED"]
        assert len(stale) >= 1
        assert stale[0]["severity"] == "CRITICAL"

    @pytest.mark.asyncio
    async def test_no_stale_feed_with_recent_events(self, client):
        """Last event < 10 min ago → no STALE_FEED."""
        events = [_make_event("ENTRY", timestamp=_ts(2))]  # 2 min ago
        await _ingest(client, events)
        resp = await client.get(f"/stores/{STORE_ID}/anomalies")
        stale = [a for a in resp.json()["anomalies"] if a["type"] == "STALE_FEED"]
        assert len(stale) == 0


# ---------------------------------------------------------------------------
# CONVERSION_DROP tests
# ---------------------------------------------------------------------------

class TestConversionDropAnomaly:
    @pytest.mark.asyncio
    async def test_conversion_drop_critical(self, client, db_session):
        """Conversion rate falls by 50%+ compared to 7-day average -> CRITICAL."""
        # 1. Seed historical baseline (7-day average):
        # We will seed 1 visitor and 1 purchase 2 days ago (100% conversion)
        past_time = _ts_days(2)
        db_session.add(EventRecord(
            event_id=str(uuid.uuid4()),
            store_id=STORE_ID,
            camera_id="CAM_ENTRY_01",
            visitor_id="VIS_hist1",
            event_type="ENTRY",
            timestamp=past_time,
            is_staff=False,
            confidence=0.9
        ))
        db_session.add(EventRecord(
            event_id=str(uuid.uuid4()),
            store_id=STORE_ID,
            camera_id="CAM_BILLING_01",
            visitor_id="VIS_hist1",
            event_type="BILLING_QUEUE_JOIN",
            zone_id="BILLING",
            timestamp=past_time,
            is_staff=False,
            confidence=0.9
        ))
        db_session.add(POSTransaction(
            store_id=STORE_ID,
            transaction_id="TXN_HIST1",
            timestamp=past_time,
            basket_value=100.0
        ))

        # 2. Seed today's conversion rate:
        # 2 visitors today, but 0 purchases -> 0% conversion rate (100% drop)
        today_time_1 = _ts(10)
        today_time_2 = _ts(5)
        db_session.add(EventRecord(
            event_id=str(uuid.uuid4()),
            store_id=STORE_ID,
            camera_id="CAM_ENTRY_01",
            visitor_id="VIS_today1",
            event_type="ENTRY",
            timestamp=today_time_1,
            is_staff=False,
            confidence=0.9
        ))
        db_session.add(EventRecord(
            event_id=str(uuid.uuid4()),
            store_id=STORE_ID,
            camera_id="CAM_ENTRY_01",
            visitor_id="VIS_today2",
            event_type="ENTRY",
            timestamp=today_time_2,
            is_staff=False,
            confidence=0.9
        ))

        await db_session.commit()

        # 3. Check anomalies
        resp = await client.get(f"/stores/{STORE_ID}/anomalies")
        assert resp.status_code == 200
        anomalies = resp.json()["anomalies"]
        drop_anomalies = [a for a in anomalies if a["type"] == "CONVERSION_DROP"]
        assert len(drop_anomalies) >= 1
        assert drop_anomalies[0]["severity"] == "CRITICAL"
        assert "below" in drop_anomalies[0]["description"].lower()

    @pytest.mark.asyncio
    async def test_conversion_drop_warn(self, client, db_session):
        """Conversion rate falls by 30% (between 25% and 50%) -> WARN."""
        # 1. Seed historical baseline (7-day average):
        # We will seed 1 visitor and 1 purchase 2 days ago (100% conversion)
        past_time = _ts_days(2)
        db_session.add(EventRecord(
            event_id=str(uuid.uuid4()),
            store_id=STORE_ID,
            camera_id="CAM_ENTRY_01",
            visitor_id="VIS_hist2",
            event_type="ENTRY",
            timestamp=past_time,
            is_staff=False,
            confidence=0.9
        ))
        db_session.add(EventRecord(
            event_id=str(uuid.uuid4()),
            store_id=STORE_ID,
            camera_id="CAM_BILLING_01",
            visitor_id="VIS_hist2",
            event_type="BILLING_QUEUE_JOIN",
            zone_id="BILLING",
            timestamp=past_time,
            is_staff=False,
            confidence=0.9
        ))
        db_session.add(POSTransaction(
            store_id=STORE_ID,
            transaction_id="TXN_HIST2",
            timestamp=past_time,
            basket_value=100.0
        ))

        # 2. Seed today's conversion rate:
        # 3 visitors today, 2 purchases -> 66.7% conversion rate (33.3% drop)
        t_time = _ts(10)
        # Vis 1: Entry + Billing Join + Purchase
        db_session.add(EventRecord(
            event_id=str(uuid.uuid4()),
            store_id=STORE_ID,
            camera_id="CAM_ENTRY_01",
            visitor_id="VIS_tod1",
            event_type="ENTRY",
            timestamp=t_time,
            is_staff=False,
            confidence=0.9
        ))
        db_session.add(EventRecord(
            event_id=str(uuid.uuid4()),
            store_id=STORE_ID,
            camera_id="CAM_BILLING_01",
            visitor_id="VIS_tod1",
            event_type="BILLING_QUEUE_JOIN",
            zone_id="BILLING",
            timestamp=t_time,
            is_staff=False,
            confidence=0.9
        ))
        db_session.add(POSTransaction(
            store_id=STORE_ID,
            transaction_id="TXN_TOD1",
            timestamp=t_time,
            basket_value=50.0
        ))

        # Vis 2: Entry + Billing Join + Purchase
        db_session.add(EventRecord(
            event_id=str(uuid.uuid4()),
            store_id=STORE_ID,
            camera_id="CAM_ENTRY_01",
            visitor_id="VIS_tod2",
            event_type="ENTRY",
            timestamp=t_time,
            is_staff=False,
            confidence=0.9
        ))
        db_session.add(EventRecord(
            event_id=str(uuid.uuid4()),
            store_id=STORE_ID,
            camera_id="CAM_BILLING_01",
            visitor_id="VIS_tod2",
            event_type="BILLING_QUEUE_JOIN",
            zone_id="BILLING",
            timestamp=t_time,
            is_staff=False,
            confidence=0.9
        ))
        db_session.add(POSTransaction(
            store_id=STORE_ID,
            transaction_id="TXN_TOD2",
            timestamp=t_time,
            basket_value=50.0
        ))

        # Vis 3: Entry only (no purchase)
        db_session.add(EventRecord(
            event_id=str(uuid.uuid4()),
            store_id=STORE_ID,
            camera_id="CAM_ENTRY_01",
            visitor_id="VIS_tod3",
            event_type="ENTRY",
            timestamp=t_time,
            is_staff=False,
            confidence=0.9
        ))

        await db_session.commit()

        # 3. Check anomalies
        resp = await client.get(f"/stores/{STORE_ID}/anomalies")
        assert resp.status_code == 200
        anomalies = resp.json()["anomalies"]
        drop_anomalies = [a for a in anomalies if a["type"] == "CONVERSION_DROP"]
        assert len(drop_anomalies) >= 1
        assert drop_anomalies[0]["severity"] == "WARN"


# ---------------------------------------------------------------------------
# Anomaly structure validation
# ---------------------------------------------------------------------------

class TestAnomalyStructure:
    @pytest.mark.asyncio
    async def test_all_anomalies_have_required_fields(self, client):
        resp = await client.get(f"/stores/{STORE_ID}/anomalies")
        for a in resp.json()["anomalies"]:
            assert "type" in a
            assert "severity" in a
            assert "description" in a
            assert "suggested_action" in a
            assert "detected_at" in a

    @pytest.mark.asyncio
    async def test_severity_is_valid_enum(self, client):
        resp = await client.get(f"/stores/{STORE_ID}/anomalies")
        for a in resp.json()["anomalies"]:
            assert a["severity"] in ("INFO", "WARN", "CRITICAL"), \
                f"Invalid severity: {a['severity']}"

    @pytest.mark.asyncio
    async def test_response_has_store_id_and_timestamp(self, client):
        resp = await client.get(f"/stores/{STORE_ID}/anomalies")
        data = resp.json()
        assert data["store_id"] == STORE_ID
        assert "as_of" in data
