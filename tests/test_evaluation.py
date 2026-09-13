"""Synthetic pure evaluation, including every frozen non-model control."""

from collections import Counter
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ovp36_benchmark import evaluation
from ovp36_benchmark.dataset import hash_dataset, load_curated_cases
from ovp36_benchmark.evaluation import (
    EvaluationError, ReviewDecision, apply_review_decisions, build_human_review_items,
    evaluate_control_case, evaluate_model_output, sentence_count_by_rule,
)
from ovp36_benchmark.identity import ExecutionKey
from ovp36_benchmark.persistence import AttemptRecorded, JOURNAL_SCHEMA_VERSION, StoredAttemptResult, StoredModelResponse
from ovp36_benchmark.requests import prepare_generated_request
from ovp36_benchmark.schemas import BenchmarkCase, GenerationConfig
from helpers import case_data, parse_case

CASES = load_curated_cases(Path(__file__).resolve().parents[1] / 'data/curated')
DATASET_HASH = hash_dataset(CASES)
FINGERPRINT = 'a' * 64


def matrix(suffix):
    return next(c for c in CASES if c.id.endswith(suffix))


def sample(task='extraction'):
    data = case_data(task)
    data['contract_id'] = task
    return parse_case(data)


def request_for(case):
    return prepare_generated_request([{'role': 'user', 'content': 'Synthetic evaluation source.'}],
        case_id=case.id, task=case.task, source=case.source, model='deterministic-unit-stub',
        generation=GenerationConfig(max_tokens=4000))


def stored(raw, *, status='success', present=True, response=True, **updates):
    value = StoredModelResponse(raw_content=raw, content_present=present, finish_reason='stop',
        response_model='deterministic-unit-stub', omitted_metadata=(), usage=None) if response else None
    return StoredAttemptResult(**(dict(status=status, latency_ms=1.0, response=value,
        error_code=None, http_status=None, retryable=False) | updates))


def evaluate(case, raw, **updates):
    request = request_for(case)
    attempt = AttemptRecorded(schema_version=JOURNAL_SCHEMA_VERSION, kind='attempt',
        execution_key=ExecutionKey(run_id='b'*64, case_id=case.id,
            request_fingerprint=request.request_fingerprint, repetition_index=0),
        attempt_index=0, result=stored(raw, **updates))
    return evaluate_model_output(case, request=request, attempt=attempt,
        dataset_hash=DATASET_HASH, evaluator_fingerprint=FINGERPRINT, purpose='functional_stub')


def qa(**updates):
    return json.dumps(dict(tags=[], overall_sentiment='neutral', call_quality_score=9,
                          summary='The caller was greeted.') | updates)


def typed_case():
    data = sample().model_dump(mode='json')
    data['input']['variables'] = [dict(name=n, type=t, hint='Synthetic typed value.')
                                  for n,t in [('s','string'), ('b','boolean'), ('n','number'), ('unknown','string')]]
    data['expected']['fields'] = {
        's': dict(state='known', acceptable_values=['Blue'], superseded_values=['Red']),
        'b': dict(state='known', acceptable_values=[True]),
        'n': dict(state='known', acceptable_values=[3]),
        'unknown': dict(state='missing', missing_policy=dict(allow_absent=True, values=[None,'unknown'])),
    }
    return parse_case(data)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        for target in ('socket.getaddrinfo', 'socket.socket.connect', 'ovp36_benchmark.client.ModelClient'):
            guard = patch(target, side_effect=AssertionError('network/client forbidden'))
            mocked = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(mocked.assert_not_called)

    def test_typed_values_exact_and_missing_policy(self):
        for missing in ({}, {'unknown':None}, {'unknown':'unknown'}):
            r = evaluate(typed_case(), json.dumps(dict(s='Blue', b=True, n=3.0) | missing))
            self.assertEqual(r.score.metrics['field_exact_count'].value, 4)
            self.assertEqual(r.score.metrics['whole_object_match'].value, 1)
            self.assertEqual(r.score.status, 'complete')

    def test_wrong_missing_stale_type_and_extra_separate(self):
        r = evaluate(typed_case(), '{"s":"Red","b":1,"unknown":"made-up","extra":4}')
        m = r.score.metrics
        self.assertEqual(m['field_exact_count'].value, 0)
        self.assertEqual(m['stale_value_count'].value, 1)
        self.assertEqual(m['unexpected_key_count'].value, 1)
        self.assertEqual(m['missing_output_count'].value, 1)
        self.assertEqual(m['known_field_count'].value, 3)
        self.assertEqual(m['missing_policy_accuracy'].value, 0)
        self.assertEqual(m['type_correctness'].value, 0.5)
        self.assertEqual(r.score.status, 'complete')

    def test_strings_no_casefold_or_trimming(self):
        for value in ('blue', ' Blue', 'Blue '):
            r = evaluate(typed_case(), json.dumps(dict(s=value,b=True,n=3)))
            self.assertEqual(r.score.metrics['field_exact_count'].value, 3)

    def test_bool_is_not_numeric_and_nonfinite_rejected(self):
        for number in ('true','"3"','NaN','Infinity','-Infinity','1e999'):
            r = evaluate(typed_case(), '{"s":"Blue","b":true,"n":'+number+'}')
            self.assertEqual(r.score.metrics['field_exact_count'].value, 3)
            self.assertTrue(r.production_parse_success)
            self.assertNotIn('NaN', r.model_dump_json())
        r = evaluate(typed_case(), '{"s":"Blue","b":true,"n":1e999}')
        self.assertTrue(r.strict_json_valid)  # Preserve parser's RFC syntax assessment.

    def test_current_precedence_and_zero_denominator(self):
        data = typed_case().model_dump(mode='json')
        data['expected']['fields']['s']['superseded_values'].append('Blue')
        r = evaluate(parse_case(data), '{"s":"Blue","b":true,"n":3}')
        self.assertEqual(r.score.metrics['stale_value_count'].value, 0)
        r = evaluate(sample(), '{"name":"Alex"}')
        self.assertIsNone(r.score.metrics['missing_policy_accuracy'].value)
        self.assertEqual(r.score.metrics['missing_policy_accuracy'].denominator, 0)

    def test_unusable_output_never_earns_allowed_absence_credit(self):
        for raw in ('broken', '', None, '[]', '```json\n42\n```'):
            r = evaluate(typed_case(), raw)
            self.assertFalse(r.production_usable)
            self.assertEqual(r.score.metrics['field_exact_count'].value, 0)

    def test_fence_and_prose_keep_semantics_and_format_separate(self):
        for raw, path in [('```json\n{"name":"Alex"}\n```','fence'), ('Answer: {"name":"Alex"}','object')]:
            r = evaluate(sample(), raw)
            self.assertEqual(r.parser_path,path)
            self.assertFalse(r.strict_format_valid)
            self.assertTrue(r.production_usable)
            self.assertEqual(r.score.metrics['field_accuracy'].value, 1)

    def test_voicemail_source_decisions_and_strictness(self):
        for text, predicted, strict, correct in [
            ('CONVERSATION','CONVERSATION',True,1), ('VOICEMAIL','VOICEMAIL',True,0),
            ('VOICEMAIL CONVERSATION','CONVERSATION',False,1), ('unclear',None,False,0),
            ('It is conversation.','CONVERSATION',False,1)]:
            r = evaluate(sample('voicemail'),text)
            self.assertEqual((r.predicted_label,r.strict_format_valid,r.score.metrics['binary_accuracy'].value),
                             (predicted,strict,correct))

    def test_vm021_and_ambiguous_labelled(self):
        for text in ('CONVERSATION','VOICEMAIL','neither'):
            r = evaluate(matrix('vm-021'),text)
            self.assertEqual(r.voicemail_stratum,'unlabelled')
            self.assertIsNone(r.score.metrics['binary_accuracy'].value)
        r = evaluate(matrix('vm-019'),'CONVERSATION')
        self.assertEqual(r.voicemail_stratum,'ambiguous_labelled')
        self.assertIsNotNone(r.score.metrics['binary_accuracy'].value)

    def test_qa_tag_scope_duplicates_and_grounding(self):
        case = matrix('qa-003')
        r = evaluate(case,qa(tags=[
            dict(tag='ASSISTANT_IN_LOOP',reason='Synthetic reason.'),
            dict(tag='ASSISTANT_IN_LOOP',reason='Repeated reason.'),
            dict(tag='HEARING_ISSUES',reason='Unsupported reason.'),
            dict(tag='DEAD_AIR',reason='Requires review.')]))
        m=r.score.metrics
        self.assertEqual([m[n].value for n in ('tag_true_positive','tag_false_negative','tag_false_positive')],[1,0,1])
        self.assertEqual(m['duplicate_tag_count'].value,1)
        self.assertEqual(r.unannotated_tags,('DEAD_AIR',))
        self.assertEqual(m['annotation_scoped_precision'].value,0.5)
        self.assertTrue(any(c.kind=='tag_grounding' and c.status=='pending' for c in r.checks))
        self.assertEqual(m['annotation_coverage'].value,0.3)

    def test_qa_missing_expected_invalid_vocabulary_and_entries(self):
        r=evaluate(matrix('qa-003'),qa())
        self.assertEqual(r.score.metrics['tag_false_negative'].value,1)
        for tags in ([dict(tag='UNKNOWN',reason='Synthetic')], [dict(tag='HEARING_ISSUES')], ['bad']):
            r=evaluate(sample('qa'),qa(tags=tags))
            self.assertFalse(r.structure_valid)
            self.assertIsNone(r.score.metrics['tag_true_positive'].value)
        r=evaluate(sample('qa'),qa(tags=[dict(tag='UNKNOWN',reason='Synthetic')]))
        self.assertEqual(r.score.metrics['invalid_tag_count'].value,1)
        self.assertNotIn('UNKNOWN',r.model_dump_json())

    def test_qa_sentiment_score_and_extra_key(self):
        for score, distance in [(8,0),(9.5,0),(10,0),(7,1),(1,7)]:
            r=evaluate(sample('qa'),qa(call_quality_score=score))
            self.assertEqual(r.score.metrics['score_distance'].value,distance)
        r=evaluate(sample('qa'),qa(overall_sentiment='negative',extra='SYNTHETIC_RAW_MARKER'))
        self.assertTrue(r.structure_valid)
        self.assertEqual(r.score.metrics['sentiment_match'].value,0)
        self.assertNotIn('SYNTHETIC_RAW_MARKER',r.model_dump_json())
        for value in (True,None,'9',float('nan'),float('inf'),11):
            r=evaluate(sample('qa'),qa(call_quality_score=value))
            self.assertFalse(r.structure_valid)
            self.assertEqual(r.score.metrics['score_coverage'].value,0)
            self.assertIsNone(r.score.metrics['score_distance'].value)
        r=evaluate(sample('qa'),qa(overall_sentiment='Neutral'))
        self.assertIsNone(r.score.metrics['sentiment_match'].value)

    def test_qa_score_single_target(self):
        data=sample('qa').model_dump(mode='json')
        data['expected'].update(quality_score_range=None,quality_score_target=8.0)
        self.assertEqual(evaluate(parse_case(data),qa(call_quality_score=9)).score.metrics['score_distance'].value,1)

    def test_qa_array_missing_fields_and_defaults_not_credit(self):
        for raw in ('[]','{}','{"tags":[]}'):
            r=evaluate(sample('qa'),raw)
            self.assertTrue(r.production_parse_success)
            self.assertFalse(r.structure_valid)
        r=evaluate(sample('qa'),'{}')
        self.assertTrue(r.production_usable)
        self.assertIsNone(r.score.metrics['tag_true_positive'].value)
        self.assertIsNone(r.score.metrics['sentiment_match'].value)

    def test_sentence_diagnostic_only(self):
        for raw, count in [('',0),('One sentence.',1),('One! Two? Three.',3),
                           ('Value 3.14 is fine.',1),('Dr. Chen arrived.',2),('One... Two',2)]:
            self.assertEqual(sentence_count_by_rule(raw),count)
        for suffix,bounds in [('qs-001',(3,5)),('ns-001',(2,4))]:
            r=evaluate(matrix(suffix),'One. Two. Three.')
            self.assertEqual(r.score.metrics['sentence_range_by_rule_ok'].value,1)
            self.assertTrue(any(c.kind=='sentence_compliance' and c.status=='pending' for c in r.checks))
        r=evaluate(matrix('cs-002'),'One. Two.')
        self.assertNotIn('sentence_count_by_rule',r.score.metrics)

    def test_summary_fact_reviews_not_substrings(self):
        case=matrix('cs-026')
        r=evaluate(case,'A paraphrase requiring review.')
        self.assertEqual(sum(c.kind=='required_fact' for c in r.checks),len(case.expected.required_facts))
        self.assertTrue(all(c.status=='pending' for c in r.checks if c.method=='human'))
        self.assertTrue(any(c.kind=='correction' for c in r.checks))
        self.assertTrue(any(c.kind=='stale_current_state' for c in r.checks))
        self.assertEqual(r.critical_status,'pending')
        for raw in ('', ' \n '):
            blank=evaluate(case,raw)
            self.assertFalse(blank.production_usable)
            self.assertEqual(next(c.status for c in blank.checks if c.check_id=='nonblank'),'fail')

    def test_safe_to_omit_and_optional_behavior_different(self):
        case=next(c for c in CASES if getattr(c.expected,'safe_to_omit',()))
        r=evaluate(case,'Synthetic summary.')
        self.assertTrue(any(c.kind=='optional_inclusion' for c in r.checks))
        items=build_human_review_items(case,r,raw_output='Synthetic summary.')
        self.assertTrue(any('Omission passes' in i.question for i in items))
        case=matrix('ns-006')
        r=evaluate(case,'Synthetic summary.')
        optional=[c for c in r.checks if 'optional_behaviors' in c.categories]
        self.assertEqual(len(optional),2)
        self.assertTrue(all(c.kind=='required_fact' and c.critical for c in optional))
        self.assertEqual(sum(c.kind=='required_fact' for c in r.checks),5)

    def test_private_transient_review_view(self):
        case=matrix('ns-006')
        raw='SYNTHETIC_PRIVATE_RESPONSE'
        r=evaluate(case,raw)
        items=build_human_review_items(case,r,raw_output=raw)
        self.assertTrue(items)
        self.assertTrue(any(i.snippets for i in items))
        for item in items:
            self.assertEqual(item.candidate_output,raw)
            self.assertNotIn(raw,repr(item)+item.model_dump_json())
            if item.statement:
                self.assertNotIn(item.statement,repr(item)+item.model_dump_json())
        with self.assertRaises(EvaluationError):
            build_human_review_items(case,r,raw_output='different')

    def test_all_twenty_controls_without_client(self):
        results=[evaluate_control_case(c,dataset_hash=DATASET_HASH,evaluator_fingerprint=FINGERPRINT)
                 for c in CASES if c.exercise.kind!='model']
        self.assertEqual(len(results),20)
        self.assertEqual(Counter(r.control_status for r in results),{'pass':19,'pending':1})
        self.assertEqual(sum(r.critical for r in results),4)
        for r in results:
            self.assertEqual(r.purpose,'control')
            self.assertIsNone(r.execution_key)
        with self.assertRaises(EvaluationError):
            evaluate_control_case(matrix('ex-001'),dataset_hash=DATASET_HASH,evaluator_fingerprint=FINGERPRINT)

    def test_qa026_negative_control_review_completion(self):
        case=matrix('qa-026')
        r=evaluate_control_case(case,dataset_hash=DATASET_HASH,evaluator_fingerprint=FINGERPRINT)
        self.assertTrue(r.production_parse_success)
        self.assertTrue(r.structure_valid)
        self.assertEqual(r.score.metrics['tag_false_positive'].value,1)
        self.assertEqual(r.control_status,'pending')
        items=build_human_review_items(case,r,raw_output=case.exercise.raw_output)
        self.assertEqual(len(items),1)
        self.assertIn('unsupported',items[0].question)
        decision=ReviewDecision(evaluation_id=r.evaluation_id,check_id=items[0].check.check_id,status='pass')
        updated=apply_review_decisions(r,[decision])
        self.assertEqual(updated.control_status,'pass')
        self.assertEqual(updated.critical_status,'pass')
        self.assertEqual(r.control_status,'pending')
        self.assertEqual(updated.score.metrics,r.score.metrics)

    def test_unknown_assertion_raises(self):
        data=matrix('ex-013').model_dump(mode='json')
        data['expected']['assertions']['unknown_operation']=True
        with self.assertRaisesRegex(EvaluationError,'unknown_control_assertion'):
            evaluate_control_case(parse_case(data),dataset_hash=DATASET_HASH,evaluator_fingerprint=FINGERPRINT)

    def test_content_transport_and_unusual_stage3b_combinations(self):
        for raw,present,expected in [('text',True,'text'),('',True,'empty'),(' \n ',True,'whitespace'),
                                      (None,True,'null'),(None,False,'content_absent')]:
            self.assertEqual(evaluate(sample(),raw,present=present).content_state,expected)
        for status in ('timeout','connection_error','http_error','protocol_error'):
            r=evaluate(sample(),'SYNTHETIC_PRIVATE',status=status)
            self.assertEqual(r.transport_status,status)
            self.assertEqual(r.content_state,'text')
            self.assertEqual(r.score.status,'not_scored')
            self.assertIsNone(r.score.metrics['field_accuracy'].value)
        r=evaluate(sample(),None,response=False)
        self.assertEqual(r.content_state,'response_absent')
        self.assertEqual(r.score.status,'not_scored')
        self.assertIsNone(evaluate(sample(),'{"name":"Alex"}').usage)

    def test_transport_error_details_are_preserved(self):
        r=evaluate(sample(),None,status='http_error',response=False,error_code='http_status_503',
                   http_status=503,retryable=True)
        self.assertEqual(r.transport_error_code,'http_status_503')
        self.assertEqual(r.http_status,503)
        self.assertTrue(r.retryable)
        self.assertEqual(r.score.status,'not_scored')

    def test_review_bindings_fail_closed(self):
        r=evaluate(matrix('ns-006'),'Synthetic summary.')
        human=next(c for c in r.checks if c.method=='human')
        automatic=next(c for c in r.checks if c.method=='automatic')
        d=ReviewDecision(evaluation_id=r.evaluation_id,check_id=human.check_id,status='pass',notes='SYNTHETIC_PRIVATE_NOTE')
        self.assertNotIn(d.notes,repr(d))
        for decisions in ([d,d], [d.model_copy(update={'evaluation_id':'0'*64})],
                          [d.model_copy(update={'check_id':'unknown'})], [d.model_copy(update={'check_id':automatic.check_id})]):
            with self.assertRaises(EvaluationError): apply_review_decisions(r,decisions)
        applied=apply_review_decisions(r,[d])
        self.assertEqual(applied.evaluation_id,r.evaluation_id)
        self.assertNotEqual(applied.checks,r.checks)

    def test_unexpected_bug_propagates(self):
        with patch('ovp36_benchmark.evaluation.parsing.parse_extraction',side_effect=RuntimeError('synthetic bug')):
            with self.assertRaises(RuntimeError): evaluate(sample(),'{}')

    def test_nonfinite_unchecked_evaluation_is_rejected_not_null_filled(self):
        r=evaluate(sample(),'{"name":"Alex"}')
        metrics=dict(r.score.metrics)
        metrics['field_accuracy']=metrics['field_accuracy'].model_copy(update={'value':float('nan')})
        invalid=r.model_copy(update={'score':r.score.model_copy(update={'metrics':metrics})})
        with self.assertRaises(EvaluationError) as caught:
            apply_review_decisions(invalid,[])
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_changed_identity_and_invalid_case_are_safe_errors(self):
        r=evaluate(sample(),'SYNTHETIC_PRIVATE_RESPONSE')
        with self.assertRaises(EvaluationError) as caught:
            apply_review_decisions(r.model_copy(update={'evidence_fingerprint':'0'*64}),[])
        self.assertNotIn('SYNTHETIC_PRIVATE_RESPONSE',str(caught.exception))
        self.assertIsNone(caught.exception.__context__)
        case=sample().model_copy(update={'expected':'SYNTHETIC_PRIVATE_INVALID'})
        with self.assertRaises(EvaluationError) as caught:
            evaluate(case,'{}')
        self.assertNotIn('SYNTHETIC_PRIVATE_INVALID',str(caught.exception))
        self.assertIsNone(caught.exception.__context__)

    def test_qa_summary_review_focus_and_critical_automatic_failure(self):
        case=matrix('qa-003')
        raw=qa(tags=[{'tag':'HEARING_ISSUES','reason':'Synthetic unsupported reason.'}])
        r=evaluate(case,raw)
        self.assertEqual(r.critical_status,'fail')
        self.assertTrue(any(c.status=='pending' and c.critical for c in r.checks))
        items=build_human_review_items(case,r,raw_output=raw)
        self.assertTrue(all('summary field only' in i.question for i in items if i.check.kind=='required_fact'))

    def test_unknown_nested_default_is_harness_error(self):
        data=matrix('qa-029').model_dump(mode='json')
        data['expected']['assertions']['source_defaults']['unknown_field']=None
        with self.assertRaisesRegex(EvaluationError,'unknown_source_default'):
            evaluate_control_case(parse_case(data),dataset_hash=DATASET_HASH,evaluator_fingerprint=FINGERPRINT)


if __name__=='__main__':
    unittest.main()
