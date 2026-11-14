"""Differential and native-path coverage for alternate30 feasible ordering."""
import ast
import copy
from dataclasses import replace
import os
from pathlib import Path
import random
import subprocess
import unittest
from unittest.mock import patch
import test_r1_fairness_alternate as alternate

A = alternate.A
BASE = '20a76449b5f02b136192c62955c5cc7977803e3b'

def baseline_order():
    source = subprocess.check_output(['git', 'show', BASE + ':vllm/v1/core/sched/r1_admission.py'],
                                     cwd=Path(__file__).resolve().parents[2]).decode()
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
             node.name in ('observe_fairness_order', 'alternate_fairness_order')]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned baseline>', 'exec'), namespace)
    return namespace['alternate_fairness_order']

def row(name, pos, wait, slo, slack):
    return dict(request_id=name, position=pos, wait_s=wait, slo_s=slo,
                prediction_p90_s=slo-wait-slack, score=slack/slo)

class OrderingReview(unittest.TestCase):
    def test_default_matches_pinned_baseline(self):
        old = baseline_order()
        rng = random.Random(8401)
        for _ in range(400):
            rows=[]
            for pos in range(rng.randrange(12)):
                wait=rng.choice([0.,1.,29.999,30.,31.])
                slo=rng.choice([2.,20.,30.,100.])
                pred=rng.choice([0.,1.,10.,100.])
                rows.append(row(str(pos),pos,wait,slo,slo-wait-pred))
            before=copy.deepcopy(rows)
            for turn in ('protected','feasible'):
                self.assertEqual(A.alternate_fairness_order(rows,30.,turn), old(rows,30.,turn))
                self.assertEqual(A.alternate_fairness_order(rows,30.,turn,'normalized'),old(rows,30.,turn))
            self.assertEqual(rows,before)

    def test_only_feasible_slots_change(self):
        # Protected includes a positive-score request; infeasible must stay FCFS.
        rows=[row('p0',0,30,100,20),row('code',1,1,30,3),
              row('i0',2,1,20,-1),row('chat',3,1,2,.5),
              row('p1',4,31,100,-1),row('i1',5,1,20,-2),
              row('summary',6,1,20,3)]
        before=copy.deepcopy(rows)
        for turn in ('protected','feasible'):
            normal=A.alternate_fairness_order(rows,30,turn)
            for mode in ('absolute','critical'):
                actual=A.alternate_fairness_order(rows,30,turn,mode,1.)
                self.assertEqual(actual[0::2],normal[0::2])
                self.assertEqual([x if x.startswith(('p','i')) else 'F' for x in actual[1]],
                                 [x if x.startswith(('p','i')) else 'F' for x in normal[1]])
                self.assertEqual([x for x in actual[1] if x.startswith('p')],['p0','p1'])
                self.assertEqual(actual[1][-2:],('i0','i1'))
        self.assertEqual(rows,before)

    def test_critical_boundary_and_noncritical_normalized_ties(self):
        rows=[row('code',0,0,30,3),row('boundary',1,0,2,1),
              row('zero',2,0,2,0),row('summary',3,0,20,2),
              row('tie',4,0,2,1)]
        self.assertEqual(A.alternate_fairness_order(rows,30,'feasible','critical',1)[1],
                         ('zero','boundary','tie','code','summary'))
        self.assertEqual(A.alternate_fairness_order(rows,30,'feasible','absolute')[1],
                         ('zero','boundary','tie','summary','code'))
        for mode in ('normalized','absolute','critical'):
            for rows in ([],[row('i',0,0,2,-1)],[row('p',0,30,40,0)]):
                self.assertFalse(A.alternate_fairness_order(rows,30,'feasible',mode,1)[2])

    def test_real_scheduler_changes_only_feasible_priority(self):
        for mode, expected in [('normalized', ['b','a']), ('absolute', ['c','a']),
                               ('critical', ['c','a'])]:
            with self.subTest(mode=mode):
                h,obj,w=alternate.AlternateTests().setup_case()
                obj.r1_admission.config=replace(obj.r1_admission.config,
                    feasible_order=mode,critical_horizon_seconds=7.)
                obj.max_num_running_reqs=2
                for request in obj.waiting:
                    request.r1_request_type='chat' if request.request_id=='c' else 'code'
                obj.r1_shadow.predictor.scores={'a':1.,'b':80.,'c':1.,'d':200.}
                h.run100(obj)
                event=h.event(w)
                self.assertEqual(event['window_admitted'],expected)
                self.assertEqual([r.request_id for r in obj.waiting],
                                 [name for name in 'abcd' if name not in expected])
                self.assertEqual(event['fairness_alternate_turn_after'],'feasible')
                self.assertEqual(event['feasible_order'],mode)

    def test_explicit_config_and_invalid_inputs(self):
        env={'VLLM_R1_ADMISSION_MODE':'apply','VLLM_R1_ADMISSION_HEAD_POLICY':'type_slo',
             'VLLM_R1_ADMISSION_TIME_OBSERVE':'1','VLLM_R1_FAIRNESS_APPLY':'1',
             'VLLM_R1_FAIRNESS_ALTERNATE':'1','VLLM_R1_FAIRNESS_WAIT_LIMIT_SECONDS':'30'}
        for mode in ('normalized','absolute','critical'):
            with patch.dict(os.environ,dict(env,VLLM_R1_FEASIBLE_ORDER=mode,
                                           VLLM_R1_CRITICAL_HORIZON_SECONDS='0.5'),clear=True):
                cfg=A.AdmissionConfig.from_env()
                self.assertEqual(cfg.feasible_order,mode)
                if mode=='critical': self.assertEqual(cfg.critical_horizon_seconds,.5)
        for value in (None,'0','-1','nan','inf','bad'):
            values=dict(env,VLLM_R1_FEASIBLE_ORDER='critical')
            if value is not None: values['VLLM_R1_CRITICAL_HORIZON_SECONDS']=value
            with patch.dict(os.environ,values,clear=True),self.assertRaises(ValueError):
                A.AdmissionConfig.from_env()
        for mode,horizon in [('bad',None),('critical',None),('critical',True),('critical',float('nan')),('critical',0)]:
            with self.assertRaises(ValueError):
                A.alternate_fairness_order([],30,'feasible',mode,horizon)
        with patch.dict(os.environ,{'VLLM_R1_FEASIBLE_ORDER':'absolute'},clear=True),self.assertRaises(ValueError):
            A.AdmissionConfig.from_env()

class AbsoluteNativePaths(alternate.AlternateTests):
    def setup_case(self):
        h,obj,w=super().setup_case()
        obj.r1_admission.config=replace(obj.r1_admission.config,feasible_order='absolute')
        return h,obj,w

class CriticalNativePaths(alternate.AlternateTests):
    def setup_case(self):
        h,obj,w=super().setup_case()
        obj.r1_admission.config=replace(obj.r1_admission.config,feasible_order='critical',critical_horizon_seconds=.5)
        return h,obj,w

if __name__=='__main__': unittest.main()
