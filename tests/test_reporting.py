"""Private synthetic journals only: no client construction or network execution."""

from collections import Counter
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from ovp36_benchmark.adapters import prepare_curated_plan
from ovp36_benchmark.client import AttemptResult, ModelResponse
from ovp36_benchmark.contracts import hash_contract, load_contract
from ovp36_benchmark.dataset import hash_dataset
from ovp36_benchmark.evaluation import (
    CaseEvaluation, EvaluationError, ReviewDecision, apply_review_decisions, evaluate_control_case,
    evaluate_model_output,
)
from ovp36_benchmark.identity import ExecutionKey, canonical_json_bytes, fingerprint_execution_plan
from ovp36_benchmark.persistence import ResultJournal, build_manifest, read_events, JournalCorruptionError
from ovp36_benchmark.replay import CapturedReplayCase, ReplayContract, ReplayDataset, prepare_replay_plan
from ovp36_benchmark.requests import CapturedReplayRequest, MessageSnapshot
from ovp36_benchmark.reporting import (
    aggregate_evaluations, evaluate_run, load_review_decisions, write_evaluation_artifacts,
)
from ovp36_benchmark.schemas import EndpointConfig, GenerationConfig, RunConfig, Task
from test_evaluation import CASES, DATASET_HASH, FINGERPRINT, evaluate, matrix, qa, sample


def config():
    return RunConfig(endpoint=EndpointConfig(alias='synthetic-evaluation',base_url_env='UNUSED_TEST_URL',
        model='deterministic-unit-stub'),generation=GenerationConfig(max_tokens=4000),
        experiment_label='evaluation-unit',evaluation='exploratory')


def manifest(cases, plan, digest=None):
    return build_manifest(config(),resolved_model='deterministic-unit-stub',dataset_hash=digest or hash_dataset(cases),
        ordered_execution_plan_hash=fingerprint_execution_plan([p.identity_projection() for p in plan]),
        contract_hashes={c.contract_id:hash_contract(load_contract(c.contract_id)) for c in cases},
        harness_code_fingerprint='c'*64,dependency_lock_fingerprint='d'*64)


def outcome(raw, status='success'):
    return AttemptResult(status=status,latency_ms=2.5,
        response=ModelResponse(raw_content=raw,content_present=True,finish_reason='stop',
                               response_model='deterministic-unit-stub',usage=None))


@unittest.skipUnless(os.name=='posix','private local POSIX artifacts')
class ReportingTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve()/'results'
        self.cases=(matrix('ex-001'),matrix('qs-001'))
        self.plan=prepare_curated_plan(self.cases,config=config(),resolved_model='deterministic-unit-stub')
        self.manifest=manifest(self.cases,self.plan)
        self.journal_path=self.root/self.manifest.run_id/'journal.jsonl'
        with ResultJournal.open(self.root,self.manifest): pass
        for target in ('socket.getaddrinfo','socket.socket.connect','ovp36_benchmark.client.ModelClient'):
            guard=patch(target,side_effect=AssertionError('network/client forbidden'))
            mocked=guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(mocked.assert_not_called)

    def key(self,index=0):
        return ExecutionKey(run_id=self.manifest.run_id,**self.plan[index].identity_projection())

    def seed(self,index=0,raw='{"destination":"Paris"}',*,finalized=True,retry=False):
        with ResultJournal.open(self.root,self.manifest) as journal:
            if retry: journal.append_attempt(self.key(index),0,outcome('earlier','timeout'))
            journal.append_attempt(self.key(index),int(retry),outcome(raw))
            if finalized: journal.finalize(self.key(index),int(retry))

    def run_evaluation(self,**updates):
        return evaluate_run(**(dict(dataset=self.cases,plan=self.plan,results_root=self.root,
            expected_manifest=self.manifest,evaluator_fingerprint=FINGERPRINT,purpose='functional_stub')|updates))

    def assert_safe(self,error,*markers):
        for marker in markers:
            self.assertNotIn(marker,str(error)+repr(error))
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)

    def test_finalized_attempt_selected_and_plan_order(self):
        self.seed(1,'Synthetic summary.',retry=True)
        self.seed(0)
        r=self.run_evaluation()
        self.assertEqual([v.case_id for v in r],[c.id for c in self.cases])
        self.assertEqual([v.attempt_index for v in r],[0,1])
        self.assertEqual(r[1].transport_status,'success')

    def test_missing_and_unfinalized_not_scored(self):
        self.seed(finalized=False)
        r=self.run_evaluation()
        self.assertEqual([v.execution_state for v in r],['unfinalized','not_attempted'])
        self.assertTrue(all(v.score.status=='not_scored' for v in r))
        self.assertTrue(all(v.production_parse_success is None for v in r))
        self.assertEqual(len(read_events(self.root,self.manifest.run_id)),1)

    def test_manifest_dataset_and_plan_mismatches(self):
        bad=self.manifest.model_copy(update={'dataset_hash':'f'*64})
        with self.assertRaisesRegex(EvaluationError,'manifest_mismatch'):
            self.run_evaluation(expected_manifest=bad)
        with self.assertRaisesRegex(EvaluationError,'dataset_mismatch'):
            self.run_evaluation(dataset=self.cases[:1])
        for plan in (self.plan[::-1],self.plan[:1],self.plan+self.plan[:1]):
            with self.assertRaisesRegex(EvaluationError,'plan_hash_mismatch'):
                self.run_evaluation(plan=plan)

    def test_foreign_execution_rejected(self):
        foreign=self.key().model_copy(update={'case_id':'synthetic-foreign'})
        with ResultJournal.open(self.root,self.manifest) as journal:
            journal.append_attempt(foreign,0,outcome('Synthetic'))
            journal.finalize(foreign,0)
        with self.assertRaisesRegex(EvaluationError,'foreign_execution'):
            self.run_evaluation()

    def test_corruption_no_partial_evaluation_or_tail_repair(self):
        self.seed()
        with self.journal_path.open('ab') as stream: stream.write(b'{"SYNTHETIC_PRIVATE":')
        before=self.journal_path.read_bytes()
        with patch('ovp36_benchmark.reporting.evaluate_model_output',side_effect=AssertionError('partial scoring')) as scoring:
            with self.assertRaises(JournalCorruptionError): self.run_evaluation()
            scoring.assert_not_called()
        self.assertEqual(self.journal_path.read_bytes(),before)

    def test_evaluation_repeat_does_not_mutate_evidence(self):
        self.seed()
        before=self.journal_path.read_bytes()
        stat_before=self.journal_path.stat()
        first=self.run_evaluation()
        self.assertEqual(first,self.run_evaluation())
        self.assertEqual(self.journal_path.read_bytes(),before)
        self.assertEqual(self.journal_path.stat().st_mtime_ns,stat_before.st_mtime_ns)

    def test_identity_binds_evidence_evaluator_purpose_case(self):
        r=evaluate(sample(),'first')
        self.assertNotEqual(r.evaluation_id,evaluate(sample(),'second').evaluation_id)
        self.seed()
        original=self.run_evaluation()[0]
        changed=self.run_evaluation(evaluator_fingerprint='f'*64)[0]
        self.assertNotEqual(original.evaluation_id,changed.evaluation_id)
        self.assertNotEqual(original.evaluation_id,self.run_evaluation(purpose='candidate')[0].evaluation_id)
        with patch('ovp36_benchmark.evaluation.EVALUATOR_VERSION','ovp36-scorer-v2'):
            self.assertNotEqual(original.evaluation_id,self.run_evaluation()[0].evaluation_id)
        self.assertNotEqual(evaluate(matrix('vm-001'),'CONVERSATION').evaluation_id,
                            evaluate(matrix('vm-002'),'CONVERSATION').evaluation_id)

    def test_aggregation_rates_unavailable_and_controls_separate(self):
        a=evaluate(matrix('vm-001'),'CONVERSATION')
        b=evaluate(matrix('vm-002'),'VOICEMAIL')
        c=evaluate(matrix('vm-003'),None,status='timeout')
        control=evaluate_control_case(matrix('vm-022'),dataset_hash=DATASET_HASH,evaluator_fingerprint=FINGERPRINT)
        report=aggregate_evaluations([a,b,c,control])
        group=report['contracts']['functional_stub:synthetic:voicemail']
        metric=group['metrics']['binary_accuracy']
        self.assertEqual((metric['numerator'],metric['denominator'],metric['unavailable_records']),(1,2,1))
        self.assertEqual(group['selected'],3)
        self.assertEqual(report['controls']['response_contract']['selected'],1)
        self.assertNotIn('overall_score',report)

    def test_voicemail_null_label_and_no_decision_denominators(self):
        rows=[evaluate(matrix('vm-001'),'neither'),evaluate(matrix('vm-021'),'neither'),
              evaluate(matrix('vm-019'),'CONVERSATION')]
        group=aggregate_evaluations(rows)['contracts']['functional_stub:synthetic:voicemail']
        self.assertEqual(group['metrics']['binary_accuracy']['denominator'],2)
        self.assertIsNone(group['voicemail']['unlabelled']['binary_accuracy'])
        self.assertEqual(group['voicemail']['labelled']['classes']['CONVERSATION']['fn'],1)
        self.assertEqual(group['voicemail']['labelled']['no_decision_rate']['value'],1)

    def test_qa_annotation_scope_macro_support_and_pending(self):
        r=evaluate(matrix('qa-003'),qa(tags=[{'tag':'ASSISTANT_IN_LOOP','reason':'Synthetic reason.'},
                                         {'tag':'DEAD_AIR','reason':'Review this.'}]))
        g=aggregate_evaluations([r])['contracts']['functional_stub:synthetic:qa']
        self.assertEqual(g['annotation_scoped_tags']['scope'],'annotation_scoped')
        self.assertEqual(g['metrics']['unannotated_tag_count']['value'],1)
        self.assertEqual(g['metrics']['annotation_coverage']['value'],0.3)
        self.assertGreater(g['checks']['tag_grounding']['pending'],0)
        self.assertIsNone(g['annotation_scoped_tags']['per_tag']['USER_FRUSTRATED']['precision']['value'])

    def test_critical_counts_and_pending_ids_separate(self):
        model=evaluate(matrix('ns-006'),'Synthetic summary.')
        control=evaluate_control_case(matrix('qa-026'),dataset_hash=DATASET_HASH,evaluator_fingerprint=FINGERPRINT)
        report=aggregate_evaluations([model,control])
        self.assertEqual(report['critical_models']['selected'],1)
        self.assertEqual(report['critical_controls']['selected'],1)
        self.assertIn(model.case_id,report['critical_models']['pending_case_ids'])
        self.assertIn(control.case_id,report['critical_controls']['pending_case_ids'])
        self.assertNotIn(control.case_id,report['critical_models']['pending_case_ids'])
        decision=ReviewDecision(evaluation_id=model.evaluation_id,check_id='required_fact.0',status='fail')
        updated=aggregate_evaluations([apply_review_decisions(model,[decision])])
        self.assertEqual(updated['critical_models']['fail'],1)
        self.assertTrue(updated['critical_models']['pending_checks'])

    def test_aggregate_rejects_duplicate_and_mixed_basis(self):
        r=evaluate(sample(),'{}')
        for rows in ([r,r],[r,r.model_copy(update={'evaluation_id':'0'*64,'dataset_hash':'0'*64})]):
            with self.assertRaises(EvaluationError): aggregate_evaluations(rows)

    def test_historical_parse_only_and_no_adapter(self):
        messages=MessageSnapshot([{'role':'user','content':'SYNTHETIC_PRIVATE_REPLAY'}])
        capture=CapturedReplayRequest(case_id='ovp34-0000000000000001',observation_id='0000000000000001',
            source_ref='synthetic-replay',task=Task.QA,messages=messages,captured_message_fingerprint=messages.fingerprint)
        case=CapturedReplayCase(contract=ReplayContract.QA_EVALUATION,captured=capture)
        dataset=ReplayDataset(cases=(case,),source_artifact_hash='1'*64,selection_artifact_hash='2'*64)
        plan=prepare_replay_plan(dataset,config=config(),resolved_model='deterministic-unit-stub')
        expected=manifest((),plan,dataset.dataset_hash)
        key=ExecutionKey(run_id=expected.run_id,**plan[0].identity_projection())
        with ResultJournal.open(self.root,expected) as journal:
            journal.append_attempt(key,0,outcome(qa()))
            journal.finalize(key,0)
        with patch('ovp36_benchmark.adapters.prepare_curated_request',side_effect=AssertionError('adapter forbidden')):
            rows=self.run_evaluation(dataset=dataset,plan=plan,expected_manifest=expected)
        self.assertTrue(rows[0].production_parse_success)
        self.assertTrue(rows[0].structure_valid)
        self.assertFalse(rows[0].gold_available)
        self.assertEqual(rows[0].score.status,'not_scored')
        self.assertIsNone(rows[0].score.metrics['sentiment_match'].value)
        self.assertNotIn('SYNTHETIC_PRIVATE_REPLAY',rows[0].model_dump_json())

    def test_artifact_create_only_private_finite_no_raw(self):
        self.seed(raw='SYNTHETIC_PRIVATE_CANDIDATE')
        rows=self.run_evaluation()
        journal_before=self.journal_path.read_bytes()
        batch=write_evaluation_artifacts(self.root,evaluations=rows)
        automatic=(batch/'automatic.jsonl').read_bytes()
        self.assertNotIn(b'SYNTHETIC_PRIVATE_CANDIDATE',automatic)
        for line in automatic.splitlines():
            CaseEvaluation.model_validate_json(line)
            self.assertNotIn(b'raw_content',line)
        for path in (batch,batch/'automatic.jsonl',batch/'summary-automatic.json'):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode)&0o077,0)
        before=(batch/'automatic.jsonl').stat().st_mtime_ns
        self.assertEqual(write_evaluation_artifacts(self.root,evaluations=rows),batch)
        self.assertEqual((batch/'automatic.jsonl').stat().st_mtime_ns,before)
        self.assertEqual(self.journal_path.read_bytes(),journal_before)
        (batch/'automatic.jsonl').write_bytes(b'conflicting synthetic content')
        with self.assertRaises(EvaluationError): write_evaluation_artifacts(self.root,evaluations=rows)

    def test_review_immutable_complete_revisions_and_private_notes(self):
        self.seed(1,'Synthetic summary.')
        rows=self.run_evaluation()
        r=rows[1]
        check=next(c for c in r.checks if c.method=='human')
        decision=ReviewDecision(evaluation_id=r.evaluation_id,check_id=check.check_id,status='pass',notes='SYNTHETIC_PRIVATE_NOTE')
        batch=write_evaluation_artifacts(self.root,evaluations=rows,review_decisions=[decision])
        automatic=(batch/'automatic.jsonl').read_bytes()
        first=list((batch/'reviews').glob('*.jsonl'))[0]
        self.assertEqual(load_review_decisions(first,evaluations=rows),(decision,))
        changed=decision.model_copy(update={'status':'fail'})
        write_evaluation_artifacts(self.root,evaluations=rows,review_decisions=[changed])
        self.assertEqual(len(list((batch/'reviews').glob('*.jsonl'))),2)
        self.assertEqual((batch/'automatic.jsonl').read_bytes(),automatic)
        for summary in batch.glob('summary*.json'):
            self.assertNotIn('SYNTHETIC_PRIVATE_NOTE',summary.read_text())
        with self.assertRaises(EvaluationError):
            write_evaluation_artifacts(self.root,evaluations=[apply_review_decisions(r,[decision])])

    def test_review_reader_fail_closed_private_errors(self):
        self.seed(1,'Synthetic summary.')
        rows=self.run_evaluation()
        path=self.root/'review.jsonl'
        for content in (b'{"SYNTHETIC_PRIVATE":',b'{"x":NaN}\n',b'{}\n',b'\n',b'{}',
                        b'{"x":1,"x":2}\n',b'\xff\n'):
            path.write_bytes(content)
            path.chmod(0o600)
            with self.assertRaises(EvaluationError) as caught: load_review_decisions(path,evaluations=rows)
            self.assert_safe(caught.exception,'SYNTHETIC_PRIVATE',str(path))

    def test_review_reader_rejects_bad_bindings(self):
        self.seed(1,'Synthetic summary.')
        rows=self.run_evaluation()
        r=rows[1]
        human=next(c for c in r.checks if c.method=='human')
        d=ReviewDecision(evaluation_id=r.evaluation_id,check_id=human.check_id,status='pass')
        path=self.root/'review.jsonl'
        for decisions in ([d,d],[d.model_copy(update={'evaluation_id':'0'*64})],
                          [d.model_copy(update={'check_id':'unknown'})],[d.model_copy(update={'check_id':'nonblank'})]):
            path.write_bytes(b''.join(canonical_json_bytes(x.model_dump(mode='json'))+b'\n' for x in decisions))
            path.chmod(0o600)
            with self.assertRaises(EvaluationError): load_review_decisions(path,evaluations=rows)

    def test_artifact_symlink_and_permission_rejection(self):
        self.seed()
        rows=self.run_evaluation()
        batch=write_evaluation_artifacts(self.root,evaluations=rows)
        for path in (batch,batch/'automatic.jsonl'):
            mode=stat.S_IMODE(path.stat().st_mode)
            for bit in (0o040,0o020,0o010,0o004,0o002,0o001):
                path.chmod(mode|bit)
                try:
                    with self.assertRaises(EvaluationError): write_evaluation_artifacts(self.root,evaluations=rows)
                finally: path.chmod(mode)
        target=self.root/'synthetic-target'
        target.write_text('Synthetic target')
        target.chmod(0o600)
        path=batch/'automatic.jsonl'
        path.unlink()
        path.symlink_to(target)
        with self.assertRaises(EvaluationError): write_evaluation_artifacts(self.root,evaluations=rows)
        self.assertEqual(target.read_text(),'Synthetic target')

    def test_artifact_fsync_failure_safe_and_no_overwrite(self):
        self.seed()
        rows=self.run_evaluation()
        with patch('ovp36_benchmark.reporting.os.fsync',side_effect=OSError('SYNTHETIC_PRIVATE_PATH')):
            with self.assertRaises(EvaluationError) as caught: write_evaluation_artifacts(self.root,evaluations=rows)
        self.assert_safe(caught.exception,'SYNTHETIC_PRIVATE_PATH')

    def test_control_artifact_separate_from_run(self):
        rows=[evaluate_control_case(c,dataset_hash=DATASET_HASH,evaluator_fingerprint=FINGERPRINT)
              for c in CASES if c.exercise.kind!='model']
        batch=write_evaluation_artifacts(self.root,evaluations=rows)
        self.assertEqual(batch.parent.name,'control-evaluations')
        summary=json.loads((batch/'summary-automatic.json').read_bytes())
        self.assertEqual(summary['contracts'],{})
        self.assertEqual(sum(v['selected'] for v in summary['controls'].values()),20)

    def test_empty_review_revision_and_publication_durability(self):
        self.seed()
        rows=self.run_evaluation()
        actions=[]
        real_sync,real_link=os.fsync,os.link
        def sync(fd):
            actions.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
            real_sync(fd)
        def link(*args,**kwargs):
            actions.append('publish')
            return real_link(*args,**kwargs)
        with patch('ovp36_benchmark.reporting.os.fsync',side_effect=sync), \
             patch('ovp36_benchmark.reporting.os.link',side_effect=link):
            batch=write_evaluation_artifacts(self.root,evaluations=rows,review_decisions=[])
        for index,action in enumerate(actions):
            if action=='publish':
                self.assertEqual(actions[index-1],'file')
                self.assertEqual(actions[index+1],'directory')
        self.assertEqual(actions.count('publish'),4)
        review=list((batch/'reviews').glob('*.jsonl'))[0]
        self.assertEqual(load_review_decisions(review,evaluations=rows),())

    def test_review_reader_rejects_symlink(self):
        target=self.root/'synthetic-review.jsonl'
        target.write_bytes(b'')
        target.chmod(0o600)
        link=self.root/'review-link.jsonl'
        link.symlink_to(target)
        with self.assertRaises(EvaluationError):
            load_review_decisions(link,evaluations=self.run_evaluation())

    def test_unexpected_reporting_bug_raises(self):
        with patch('ovp36_benchmark.reporting.aggregate_evaluations',side_effect=RuntimeError('synthetic bug')):
            with self.assertRaises(RuntimeError): write_evaluation_artifacts(self.root,evaluations=self.run_evaluation())


if __name__=='__main__':
    unittest.main()
