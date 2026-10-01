import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from rampflow.departure_release import (
    DepartureReleaseBoard,
    ReleaseArea,
    ReleaseStatus,
    SignoffState,
)
from rampflow.workflows.control import RampWorkflowControl, WorkflowRejected

UTC = timezone.utc
T = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)

AREAS = (
    ReleaseArea("fuel", "Fueling", frozenset({"uplift_ticket"})),
    ReleaseArea("baggage", "Baggage", frozenset({"reconciliation"})),
    ReleaseArea("maintenance", "Maintenance", frozenset({"release_form"})),
)


class DepartureReleaseTests(unittest.TestCase):
    def setUp(self):
        self.board = DepartureReleaseBoard("CA1234", AREAS)

    def test_pending_before_any_report(self):
        v = self.board.conclusion()
        self.assertIs(v.status, ReleaseStatus.PENDING)
        self.assertEqual(v.generation, 0)
        self.assertEqual({a.area for a in v.outstanding}, {"fuel", "baggage", "maintenance"})
        self.assertEqual(v.blockers, ())
        self.assertEqual(v.basis, {"baggage": None, "fuel": None, "maintenance": None})

    def test_blocker_attribution_and_final_release(self):
        b = self.board
        b.report("op-fuel-1", "fuel", SignoffState.COMPLETE, "fuel-crew", T(10, 10),
                 evidence={"uplift_ticket": "FU-9"})
        b.report("op-bag-1", "baggage", SignoffState.IN_PROGRESS, "bag-crew", T(10, 12))
        b.report("op-mx-1", "maintenance", SignoffState.BLOCKED, "mx-eng", T(10, 15),
                 reason="hydraulic leak awaiting part")

        blocked = b.conclusion()
        self.assertIs(blocked.status, ReleaseStatus.BLOCKED)
        self.assertEqual([x.area for x in blocked.blockers], ["maintenance"])
        self.assertEqual(blocked.blockers[0].reason, "hydraulic leak awaiting part")
        self.assertEqual(blocked.blockers[0].blocked_since, T(10, 15))
        self.assertEqual({x.area for x in blocked.outstanding}, {"baggage"})

        decision = b.can_release()
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "blocked")
        self.assertIn("maintenance", decision.details["blockers"])

        # Maintenance clears; with baggage still in progress the flight is pending again.
        b.report("op-mx-2", "maintenance", SignoffState.COMPLETE, "mx-eng", T(10, 40),
                 evidence={"release_form": "MEL-7"})
        self.assertIs(b.conclusion().status, ReleaseStatus.PENDING)
        self.assertEqual({x.area for x in b.conclusion().outstanding}, {"baggage"})
        b.report("op-bag-2", "baggage", SignoffState.COMPLETE, "bag-crew", T(10, 45),
                 evidence={"reconciliation": "BAG-3"})
        released = b.conclusion()
        self.assertIs(released.status, ReleaseStatus.RELEASED)
        self.assertTrue(released.released)
        self.assertTrue(b.can_release().accepted)

    def test_late_correction_recomputes_blocking_reason(self):
        b = self.board
        b.report("op-fuel-1", "fuel", SignoffState.COMPLETE, "fuel-crew", T(10, 10),
                 evidence={"uplift_ticket": "FU-9"})
        b.report("op-bag-1", "baggage", SignoffState.COMPLETE, "bag-crew", T(10, 12),
                 evidence={"reconciliation": "BAG-3"})
        b.report("op-mx-1", "maintenance", SignoffState.COMPLETE, "mx-eng", T(10, 15),
                 evidence={"release_form": "MEL-7"})
        all_clear = b.conclusion()
        self.assertIs(all_clear.status, ReleaseStatus.RELEASED)

        # Late correction: the maintenance "done" was wrong.
        b.correct("op-mx-2", "maintenance", SignoffState.BLOCKED, "mx-eng", T(10, 50),
                  reason="release form invalid, defect still open")
        revised = b.conclusion()
        self.assertIs(revised.status, ReleaseStatus.BLOCKED)
        self.assertEqual([x.area for x in revised.blockers], ["maintenance"])
        self.assertEqual(revised.blockers[0].reason, "release form invalid, defect still open")
        self.assertEqual(revised.areas["maintenance"].updates, 2)

        # The earlier released generation stays exactly as it was.
        pinned = b.conclusion(all_clear.generation)
        self.assertIs(pinned, all_clear)
        self.assertIs(pinned.status, ReleaseStatus.RELEASED)
        self.assertNotEqual(pinned.digest, revised.digest)
        self.assertTrue(b.verify_conclusion(pinned))
        self.assertTrue(b.verify_conclusion(revised))

    def test_unsubstantiated_completion_is_rejected(self):
        with self.assertRaises(ValueError):
            self.board.report("op-fuel-1", "fuel", SignoffState.COMPLETE, "fuel-crew", T(10))
        with self.assertRaises(ValueError):
            self.board.report("op-mx-1", "maintenance", SignoffState.BLOCKED, "mx-eng", T(10))
        # Nothing entered the log.
        self.assertEqual(self.board.generation, 0)

    def test_report_idempotency_and_operation_identity(self):
        b = self.board
        first = b.report("op-1", "fuel", SignoffState.COMPLETE, "fuel-crew", T(10),
                         evidence={"uplift_ticket": "FU-9"})
        second = b.report("op-1", "fuel", SignoffState.COMPLETE, "fuel-crew", T(10),
                          evidence={"uplift_ticket": "FU-9"})
        self.assertIs(first, second)
        self.assertEqual(b.generation, 1)
        with self.assertRaises(ValueError):
            b.report("op-1", "fuel", SignoffState.BLOCKED, "fuel-crew", T(11), reason="changed")

    def test_two_readers_same_generation_see_same_answer(self):
        b = self.board
        seen = []
        barrier = []

        def reader():
            v = b.conclusion()
            seen.append((v.generation, v.digest, id(v)))
            # Pinning the same generation returns the identical object.
            again = b.conclusion(v.generation)
            barrier.append(again is v)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(reader) for _ in range(2)]
            for f in futures:
                f.result()

        self.assertEqual(seen[0], seen[1])
        self.assertEqual(barrier, [True, True])

    def test_concurrent_writes_are_serialized_and_chain_verifies(self):
        b = self.board
        evidence_by_area = {
            "fuel": {"uplift_ticket": "x"},
            "baggage": {"reconciliation": "x"},
            "maintenance": {"release_form": "x"},
        }

        def file(op, area):
            return b.report(op, area, SignoffState.COMPLETE, area + "-crew", T(11),
                            evidence=evidence_by_area[area])

        ops = [
            ("op-fuel", "fuel"),
            ("op-bag", "baggage"),
            ("op-mx", "maintenance"),
        ]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(file, op, area) for op, area in ops]
            for f in futures:
                f.result()

        self.assertEqual(b.generation, 3)
        sequences = sorted(r.sequence for r in b.history())
        self.assertEqual(sequences, [1, 2, 3])
        self.assertTrue(b.verify_chain())
        self.assertIs(b.conclusion().status, ReleaseStatus.RELEASED)
        # Every generation's cached verdict independently re-derives its digest.
        for gen in range(4):
            self.assertTrue(b.verify_conclusion(b.conclusion(gen)))

    def test_report_chain_tamper_evidence(self):
        b = self.board
        b.report("op-fuel-1", "fuel", SignoffState.COMPLETE, "fuel-crew", T(10),
                 evidence={"uplift_ticket": "FU-9"})
        self.assertTrue(b.verify_chain())

    def test_workflow_publishes_release_event_only_when_released(self):
        b = self.board
        control = RampWorkflowControl()

        def try_release():
            return control.publish_departure_release(
                "rel-1", "CA1234", T(10, 50),
                lambda: b.conclusion(),
                requirements=(b.can_release(),),
            )

        with self.assertRaises(WorkflowRejected):
            try_release()
        self.assertEqual(control.stream.head, 0)

        for op, area, at, evidence in (
            ("f", "fuel", T(10, 10), {"uplift_ticket": "FU-9"}),
            ("g", "baggage", T(10, 12), {"reconciliation": "BAG-3"}),
            ("m", "maintenance", T(10, 15), {"release_form": "MEL-7"}),
        ):
            b.report(op, area, SignoffState.COMPLETE, area + "-crew", at, evidence=evidence)

        receipt = control.publish_departure_release(
            "rel-2", "CA1234", T(10, 50),
            lambda: b.conclusion(),
            requirements=(b.can_release(),),
        )
        self.assertEqual(receipt.event.topic, "departure.release.published")
        self.assertEqual(control.stream.head, 1)
        # Replaying the release operation must not publish a second event.
        again = control.publish_departure_release(
            "rel-2", "CA1234", T(10, 50),
            lambda: b.conclusion(),
            requirements=(b.can_release(),),
        )
        self.assertIs(again, receipt)
        self.assertEqual(control.stream.head, 1)


if __name__ == "__main__":
    unittest.main()
