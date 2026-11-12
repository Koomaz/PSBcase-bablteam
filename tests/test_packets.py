"""Packet invariants independent of generator quality."""
import copy
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ML1"))
sys.path.insert(0, str(ROOT / "ML2"))
from retrieval import Retriever, Settings
from llmguard.evidence import build_evidence
from llmguard.prompt_builder import render_evidence


class PacketChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = Retriever.from_artifacts(ROOT / "ML1/artifacts", settings=Settings())
        cls.lookup = {x.chunk_id: x for x in cls.r._index.chunks}

    def test_sources_are_exact_and_packet_does_not_mutate_ranking(self):
        q = "Какая гарантия по П-17/2024?"
        hits, diag = self.r.search_with_diagnostics(q)
        previous = copy.deepcopy(hits)
        expanded, new_diag = self.r.evidence_packet(q, hits, diag)
        self.assertGreater(len(expanded), len(hits))
        self.assertEqual([x.to_dict() for x in hits], [x.to_dict() for x in previous])
        self.assertNotIn("packet_route", diag)
        for item in build_evidence(expanded):
            self.assertEqual(item["text"], self.lookup[item["chunk_id"]].text.strip())
            self.assertIn("doc_id", item["meta"])
        self.assertLessEqual(len(render_evidence(build_evidence(expanded))), 7000)

    def test_no_history_leaks_between_questions(self):
        self.r.search_with_diagnostics("Какая гарантия?", history=[{"role":"user","content":"Договор П-17/2024"}])
        q = "Как испечь торт?"
        hits, diag = self.r.search_with_diagnostics(q)
        packet, _ = self.r.evidence_packet(q, hits, diag)
        self.assertEqual(packet, [])

    def test_historical_regulation_metadata_is_preserved(self):
        q = "По регламенту в феврале 2025 года кто согласует договор на 12 миллионов рублей?"
        hits, diag = self.r.search_with_diagnostics(q)
        packet, _ = self.r.evidence_packet(q, hits, diag)
        old = [x for x in packet if x.metadata.get("edition_date") == "2023-01-10"]
        self.assertTrue(any(x.metadata.get("clause") == "2.1" for x in old))
        for x in old:
            self.assertIn("temporal_status", x.metadata)

    def test_baseline_switch_preserves_original_objects(self):
        q = "Какая гарантия по П-17/2024?"
        hits, diag = self.r.search_with_diagnostics(q)
        with patch.dict(os.environ, {"PSB_EVIDENCE_MODE": "baseline"}):
            packet, _ = self.r.evidence_packet(q, hits, diag)
        self.assertIs(packet, hits)


if __name__ == "__main__":
    unittest.main()
