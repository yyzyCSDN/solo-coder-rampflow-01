import threading
import unittest
from dataclasses import replace
from datetime import datetime, timezone

from rampflow.departure_readiness import (
    DepartureReadinessBoard,
    DepartureVerdict,
    ReadinessClaim,
    ReadinessStatus,
)

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)
DEPTS = ("fuel", "baggage", "maintenance")


def claim(flight, dept, seq, status, reason="", at=10):
    return ReadinessClaim(flight, dept, seq, status, reason, D(at), D(at))


class DepartureReadinessTests(unittest.TestCase):
    def setUp(self):
        self.board = DepartureReadinessBoard(DEPTS)

    def complete_all(self, flight="F1"):
        for dept in DEPTS:
            d = self.board.submit(claim(flight, dept, 1, ReadinessStatus.COMPLETE))
            self.assertTrue(d.accepted)

    def test_ready_only_when_all_required_complete(self):
        self.complete_all()
        a = self.board.assess("F1")
        self.assertEqual(a.verdict, DepartureVerdict.READY)
        self.assertEqual(a.blockers, ())
        self.assertEqual(a.generation, 3)
        self.assertTrue(a.verify())

    def test_blockers_name_who_is_stuck_and_why(self):
        self.board.submit(claim("F1", "fuel", 1, ReadinessStatus.COMPLETE))
        self.board.submit(claim("F1", "baggage", 1, ReadinessStatus.BLOCKED, "等行李拖车"))
        a = self.board.assess("F1")
        self.assertEqual(a.verdict, DepartureVerdict.BLOCKED)
        self.assertEqual(
            [(b.department, b.kind, b.reason) for b in a.blockers],
            [
                ("baggage", "reported-blocked", "等行李拖车"),
                ("maintenance", "awaiting-confirmation", ""),
            ],
        )

    def test_pending_until_everyone_reports(self):
        self.board.submit(claim("F1", "fuel", 1, ReadinessStatus.IN_PROGRESS))
        a = self.board.assess("F1")
        self.assertEqual(a.verdict, DepartureVerdict.PENDING)
        self.assertEqual({b.kind for b in a.blockers}, {"awaiting-confirmation"})

    def test_late_correction_recomputes_conclusion_and_keeps_history(self):
        self.complete_all()
        ready = self.board.assess("F1")
        self.assertEqual(ready.verdict, DepartureVerdict.READY)

        d = self.board.submit(claim("F1", "maintenance", 2, ReadinessStatus.BLOCKED, "等待航材", at=11))
        self.assertTrue(d.accepted)
        self.assertEqual(d.details["generation"], 4)

        now = self.board.assess("F1")
        self.assertEqual(now.verdict, DepartureVerdict.BLOCKED)
        self.assertEqual(
            [(b.department, b.reason) for b in now.blockers],
            [("maintenance", "等待航材")],
        )
        self.assertNotEqual(now.digest, ready.digest)

        then = self.board.assess("F1", generation=ready.generation)
        self.assertIs(then, ready)
        self.assertEqual(then.verdict, DepartureVerdict.READY)

    def test_correction_chain_tracks_latest_reason(self):
        def fuel_reason(assessment):
            return next(b.reason for b in assessment.blockers if b.department == "fuel")

        self.board.submit(claim("F1", "fuel", 1, ReadinessStatus.BLOCKED, "等油车"))
        first = self.board.assess("F1")
        self.assertEqual(fuel_reason(first), "等油车")
        self.board.submit(claim("F1", "fuel", 2, ReadinessStatus.BLOCKED, "油车故障"))
        second = self.board.assess("F1")
        self.assertEqual(fuel_reason(second), "油车故障")
        self.assertEqual(fuel_reason(self.board.assess("F1", first.generation)), "等油车")

    def test_same_generation_gives_one_answer_to_concurrent_readers(self):
        self.board.submit(claim("F1", "fuel", 1, ReadinessStatus.COMPLETE))
        self.board.submit(claim("F1", "baggage", 1, ReadinessStatus.BLOCKED, "等行李拖车"))
        gen = self.board.generation
        results = []

        def read():
            results.append(self.board.assess("F1", gen))

        threads = [threading.Thread(target=read) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), 8)
        self.assertTrue(all(r is results[0] for r in results))
        self.assertEqual(len({r.digest for r in results}), 1)

    def test_replay_of_same_claim_is_idempotent(self):
        c = claim("F1", "fuel", 1, ReadinessStatus.COMPLETE)
        first = self.board.submit(c)
        second = self.board.submit(c)
        self.assertTrue(first.accepted)
        self.assertTrue(second.accepted)
        self.assertEqual(second.code, "duplicate")
        self.assertEqual(self.board.generation, 1)
        self.assertEqual(self.board.assess("F1").generation, 1)

    def test_stale_sequence_rejected(self):
        self.board.submit(claim("F1", "fuel", 5, ReadinessStatus.IN_PROGRESS))
        d = self.board.submit(claim("F1", "fuel", 4, ReadinessStatus.COMPLETE))
        self.assertFalse(d.accepted)
        self.assertEqual(d.code, "stale-sequence")

    def test_same_sequence_different_content_rejected(self):
        self.board.submit(claim("F1", "fuel", 1, ReadinessStatus.IN_PROGRESS))
        d = self.board.submit(claim("F1", "fuel", 1, ReadinessStatus.COMPLETE))
        self.assertFalse(d.accepted)
        self.assertEqual(d.code, "sequence-conflict")

    def test_expected_generation_compare_and_swap(self):
        self.board.submit(claim("F1", "fuel", 1, ReadinessStatus.IN_PROGRESS))
        stale = self.board.submit(claim("F1", "baggage", 1, ReadinessStatus.COMPLETE), expected_generation=5)
        self.assertFalse(stale.accepted)
        self.assertEqual(stale.code, "stale-generation")
        ok = self.board.submit(claim("F1", "baggage", 1, ReadinessStatus.COMPLETE), expected_generation=1)
        self.assertTrue(ok.accepted)

    def test_future_generation_rejected_and_empty_board_pending(self):
        a = self.board.assess("F1")
        self.assertEqual(a.generation, 0)
        self.assertEqual(a.verdict, DepartureVerdict.PENDING)
        self.assertTrue(all(s.status is ReadinessStatus.SILENT for s in a.departments.values()))
        with self.assertRaises(ValueError):
            self.board.assess("F1", generation=1)

    def test_assessment_verify_detects_tampering(self):
        self.complete_all()
        a = self.board.assess("F1")
        self.assertTrue(a.verify())
        forged = replace(a, verdict=DepartureVerdict.BLOCKED)
        self.assertFalse(forged.verify())

    def test_extra_departments_inform_but_do_not_gate(self):
        board = DepartureReadinessBoard(("fuel",))
        board.submit(claim("F1", "fuel", 1, ReadinessStatus.COMPLETE))
        board.submit(claim("F1", "catering", 1, ReadinessStatus.BLOCKED, "缺餐食"))
        a = board.assess("F1")
        self.assertEqual(a.verdict, DepartureVerdict.READY)
        self.assertEqual(a.departments["catering"].status, ReadinessStatus.BLOCKED)

    def test_claims_audit_trail_is_acceptance_ordered(self):
        self.complete_all()
        self.board.submit(claim("F1", "fuel", 2, ReadinessStatus.BLOCKED, "重新加油", at=11))
        trail = self.board.claims("F1")
        self.assertEqual([r.generation for r in trail], [1, 2, 3, 4])
        self.assertEqual(trail[-1].claim.reason, "重新加油")


if __name__ == "__main__":
    unittest.main()
