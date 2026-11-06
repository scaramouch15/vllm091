"""Opt-in type SLO protection using the actual Scheduler test harness."""
from dataclasses import replace
import os
from unittest.mock import patch

import test_r1_type_slo_admission as base


class FeasibleProtectionTests(base.TypeAdmissionTests):
    def configured(self, names='abcd'):
        obj, pred, writer = self.make_type(names=names)
        obj.max_num_running_reqs = 1
        obj.r1_admission.config = replace(obj.r1_admission.config,
            feasible_protection=True, protection_slo_multiplier=1.0)
        return obj, pred, writer

    def test_protected_precedes_feasible_and_preserves_queue_order(self):
        obj, pred, writer = self.configured()
        obj.waiting[0].r1_first_queued_timestamp = 0.0
        obj.waiting[1].r1_first_queued_timestamp = 88.0
        self.run100(obj)
        row = self.event(writer)
        self.assertEqual(row['feasible_rule_branch'], 'protected')
        self.assertEqual(row['sorted_ids'][:2], ('a', 'b'))
        self.assertEqual(row['actual_admission_order'], ['a'])
        self.assertEqual(row['type_rank_candidates'][0]['feasible_protection_limit_s'], 30.)
        self.assertEqual(row['type_rank_candidates'][1]['feasible_protection_limit_s'], 10.)

    def test_feasible_precedes_negative(self):
        obj, pred, writer = self.configured()
        obj.r1_type_slo = base.types.SimpleNamespace(snapshot=lambda: {'config_seconds': {
            'chat.ttft': 2., 'chat.tpot': .1, 'code.e2el': 100., 'summary.e2el': 100.}})
        self.run100(obj)
        row = self.event(writer)
        self.assertLess(row['type_rank_candidates'][1]['score'], 0)
        self.assertEqual(row['feasible_rule_branch'], 'feasible')
        self.assertEqual(row['actual_admission_order'], ['a'])

    def test_all_infeasible_uses_original_order(self):
        obj, pred, writer = self.configured()
        pred.scores = {name: 200 for name in 'abcd'}
        for req in obj.waiting:
            req.r1_request_type = 'code'
        self.run100(obj)
        row = self.event(writer)
        self.assertEqual(row['feasible_rule_branch'], 'all_infeasible_fcfs')
        self.assertEqual(row['sorted_ids'], tuple('abcd'))
        self.assertEqual(row['actual_admission_order'], ['a'])

    def test_observe_keeps_native_and_missing_rolls_back(self):
        obj, pred, writer = self.configured()
        obj.r1_admission.config = replace(obj.r1_admission.config, mode='observe')
        self.run100(obj)
        self.assertEqual(self.event(writer)['actual_admission_order'], ['a'])
        obj, pred, writer = self.configured()
        obj.waiting[0].r1_request_type = None
        self.run100(obj)
        self.assertEqual(self.event(writer)['reason'], 'type_information_unavailable')
        self.assertEqual(self.event(writer)['actual_admission_order'], ['a'])

    def test_config_default_off_and_invalid_multiplier(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(base.base.A.AdmissionConfig.from_env().feasible_protection)
        for value in ('nan', 'inf', '0', '-1'):
            with patch.dict(os.environ, {'VLLM_R1_ADMISSION_HEAD_POLICY':'type_slo',
                    'VLLM_R1_ADMISSION_FEASIBLE_PROTECTION':'1',
                    'VLLM_R1_ADMISSION_PROTECTION_SLO_MULTIPLIER':value}, clear=True):
                with self.assertRaises(ValueError):
                    base.base.A.AdmissionConfig.from_env()

    def test_shadow_reference_and_applied_policy_are_distinct(self):
        obj, pred, writer = self.configured()
        pred.scores = {'a': 120., 'b': 200., 'c': 150., 'd': 180.}
        for req in obj.waiting:
            req.r1_request_type = 'code'
        self.run100(obj)
        admission = self.event(writer)
        reference = [row for kind, row in writer.rows if kind == 'shadow_type_rank_v1'][-1]
        self.assertEqual(admission['feasible_rule_branch'], 'all_infeasible_fcfs')
        self.assertEqual(admission['score_rule'], 'type_slo_feasible_protection_v1')
        self.assertEqual(admission['sorted_ids'], tuple('abcd'))
        self.assertEqual(reference['comparable_order_ids'], ['b', 'd', 'c', 'a'])
        self.assertEqual(admission['actual_admission_order'], ['a'])

    def test_zero_slack_is_feasible_and_stable(self):
        obj, pred, writer = self.configured()
        for req in obj.waiting:
            req.r1_request_type = 'code'
        pred.scores = {'a': 200., 'b': 99., 'c': 99., 'd': 150.}
        self.run100(obj)
        admission = self.event(writer)
        self.assertEqual(admission['feasible_rule_branch'], 'feasible')
        self.assertEqual(admission['sorted_ids'], ('b', 'c', 'a', 'd'))
        self.assertEqual(admission['actual_admission_order'], ['b'])
