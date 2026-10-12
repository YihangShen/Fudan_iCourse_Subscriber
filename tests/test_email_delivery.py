"""Offline regressions: python -m unittest discover -s tests -v."""
import smtplib
import tempfile
import unittest
from contextlib import ExitStack
from email import policy
from email.parser import BytesParser
from pathlib import Path
from unittest.mock import Mock, patch

import main
from src.api.emailer import Emailer, _important_notices
from src.data.database import Database
from src.runtime import config


SUMMARY = r'''### 【课程事项提醒】
- **作业**：周五提交第 3 题。
#### 考试
下周一小测，请带计算器。
### 课程内容
矩阵公式：$A^{-1}$，以及
$$\sum_{i=1}^{n} x_i$$
中文、α、→ 都应保留。
'''
ITEM = {'sub_id': '123', 'course_title': '数字逻辑', 'sub_title': '第 1 节',
        'date': '2026-10-12', 'summary': SUMMARY}


class EmailFormatTests(unittest.TestCase):
    def setUp(self):
        self.mailer = Emailer()
        self.mailer.sender = 'sender@example.com'
        self.mailer.receiver = 'receiver@example.com'
        self.mailer.password = 'test-only'

    def test_unicode_mime_roundtrip_preserves_notes_and_math(self):
        with patch('src.api.emailer.requests.get', side_effect=AssertionError('No rendering network calls')):
            message = self.mailer._build_message([ITEM, dict(ITEM, sub_id='124')])
        parsed = BytesParser(policy=policy.default).parsebytes(message.as_bytes())
        self.assertFalse(parsed.defects)
        self.assertIn('数字逻辑', str(parsed['Subject']))
        body = parsed.get_body(preferencelist=('plain',)).get_content()
        self.assertIn('作业：周五提交第 3 题。', body)
        self.assertIn('下周一小测', body)
        self.assertNotIn('矩阵公式', body)
        self.assertNotIn('**', body)
        attachments = list(parsed.iter_attachments())
        self.assertEqual(len(attachments), 2)
        self.assertNotEqual(attachments[0].get_filename(), attachments[1].get_filename())
        for part in attachments:
            self.assertEqual(part.get_content_type(), 'text/markdown')
            self.assertEqual(part.get_content_charset(), 'utf-8')
            self.assertIn('数字逻辑', part.get_filename())
            self.assertTrue(part.get_filename().endswith('.md'))
            self.assertIn(SUMMARY, part.get_payload(decode=True).decode('utf-8'))
        self.assertNotIn('text/html', [part.get_content_type() for part in parsed.walk()])
        self.assertNotIn('image/png', [part.get_content_type() for part in parsed.walk()])

    def test_no_notice_does_not_claim_no_homework(self):
        self.assertEqual(_important_notices('### 数学\n作业调度算法。'),
                         '笔记未单独列出课程事项，请查阅附件确认。')

    def test_notice_heading_inside_code_is_not_a_notice(self):
        self.assertIn('请查阅附件', _important_notices('```md\n### 【课程事项提醒】\n假的作业\n```'))

    def test_legacy_bold_notice_label(self):
        text = _important_notices('**【课程事项提醒】**\n周五交作业。\n### 正文\n课程理论。')
        self.assertEqual(text, '周五交作业。')

    def test_filename_and_header_cannot_inject_newlines(self):
        item = dict(ITEM, course_title='数字\r\nBcc: bad@example.com/../逻辑')
        parsed = BytesParser(policy=policy.default).parsebytes(self.mailer._build_message([item]).as_bytes())
        self.assertIsNone(parsed['Bcc'])
        name = next(parsed.iter_attachments()).get_filename()
        self.assertNotIn('/', name)
        self.assertNotIn('\r', name)
        self.assertNotIn('\n', name)

    def test_smtp_retry_and_timeout(self):
        with patch('src.api.emailer.smtplib.SMTP_SSL', side_effect=smtplib.SMTPException('test failure')) as smtp, patch('src.api.emailer.time.sleep'):
            self.assertFalse(self.mailer.send([ITEM]))
        self.assertEqual(smtp.call_count, 3)
        self.assertEqual(smtp.call_args.kwargs['timeout'], 60)

    def test_no_duplicate_retry_after_acceptance_and_quit_error(self):
        with patch('src.api.emailer.smtplib.SMTP_SSL') as smtp:
            smtp.return_value.__enter__.return_value.send_message.return_value = {}
            smtp.return_value.__exit__.side_effect = smtplib.SMTPServerDisconnected('QUIT failed')
            self.assertTrue(self.mailer.send([ITEM]))
        self.assertEqual(smtp.call_count, 1)

    def test_missing_recipient_fails_without_smtp_connection(self):
        self.mailer.receiver = ''
        with patch('src.api.emailer.smtplib.SMTP_SSL') as smtp:
            with self.assertRaisesRegex(ValueError, 'RECEIVER_EMAIL'):
                self.mailer.send([ITEM])
        smtp.assert_not_called()


class DeliveryStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(str(Path(self.tmp.name) / 'test.db'))
        self.db.upsert_course('1', '数字逻辑', '测试老师')
        self.db.insert_lecture('123', '1', '第 1 节', '2026-10-12')
        self.db.update_summary('123', SUMMARY, 'test')
        self.db.mark_processed('123')
        self.reporter = Mock()

    def tearDown(self):
        self.db.conn.close()
        self.tmp.cleanup()

    def test_failed_delivery_remains_unsent_and_can_recover(self):
        failed = Mock()
        failed.send.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'SMTP delivery failed'):
            main._send_email(failed, self.db, self.reporter, [])
        self.assertEqual(len(self.db.get_unsent_lectures()), 1)
        success = Mock()
        success.send.return_value = True
        # Duplicate input + recovered row must be sent only once.
        main._send_email(success, self.db, self.reporter, [ITEM.copy()])
        self.assertEqual(len(success.send.call_args.args[0]), 1)
        self.assertEqual(self.db.get_unsent_lectures(), [])

    def test_missing_configuration_is_visible_and_keeps_pending_notes(self):
        with patch.object(config, 'RECEIVER_EMAIL', ''):
            with self.assertRaisesRegex(RuntimeError, 'RECEIVER_EMAIL'):
                main._send_email(None, self.db, self.reporter, [])
        self.assertEqual(len(self.db.get_unsent_lectures()), 1)

    def test_empty_queue_logs_skip_without_sending(self):
        self.db.mark_emailed_batch(['123'])
        mailer = Mock()
        main._send_email(mailer, self.db, self.reporter, [])
        mailer.send.assert_not_called()
        self.reporter.info.assert_called_with('[Email] No new or previously unsent notes; no email sent.')


class RunStatusTests(unittest.TestCase):
    def test_course_enumeration_failure_is_reported(self):
        client = Mock()
        client.get_course_detail.side_effect = RuntimeError('test API failure')
        failures = []
        with patch.object(config, 'COURSE_IDS', ['1']), patch.object(main, '_check_session'):
            self.assertEqual(main._enumerate_lectures(client, Mock(), Mock(), failures), [])
        self.assertEqual(len(failures), 1)

    def test_failed_lecture_does_not_block_next_lecture(self):
        failures, items = [], []
        lectures = [('1', '数字逻辑', {'sub_id': '1'}), ('1', '数字逻辑', {'sub_id': '2'})]
        scheduler = Mock()
        with patch.object(main, 'LectureRunner') as runner, patch.object(main, '_check_session'):
            runner.return_value.run.side_effect = [RuntimeError('model not found'), SUMMARY]
            main._drive_lectures(Mock(), Mock(), scheduler, Mock(), Mock(), Mock(), lectures, items, failures)
        self.assertEqual(len(failures), 1)
        self.assertEqual([item['sub_id'] for item in items], ['2'])
        self.assertEqual(scheduler.audio_downloader.release.call_count, 2)

    def test_transcription_failure_returning_none_is_reported(self):
        db = Mock()
        db.get_lecture.return_value = {'processed_at': None, 'error_stage': 'transcribe'}
        failures = []
        with patch.object(main, 'LectureRunner') as runner, patch.object(main, '_check_session'):
            runner.return_value.run.return_value = None
            main._drive_lectures(Mock(), db, Mock(), Mock(), Mock(), Mock(), [('1', '逻辑', {'sub_id': '1'})], [], failures)
        self.assertEqual(len(failures), 1)

    def test_run_exits_failed_but_attempts_delivery_of_completed_notes(self):
        with ExitStack() as stack:
            for name in ('Database', 'Transcriber', 'Summarizer', 'Emailer', 'Reporter',
                         'Scheduler', 'login_with_retry', 'ICourseClient', '_crawl_semester_catalog'):
                stack.enter_context(patch.object(main, name))
            stack.enter_context(patch.object(config, 'COURSE_IDS', ['1']))
            stack.enter_context(patch.object(main, '_enumerate_lectures', return_value=[]))
            def failed_pipeline(*args):
                args[-2].append(ITEM.copy())
                args[-1].append('Lecture 2: processing failed')
            stack.enter_context(patch.object(main, '_drive_lectures', side_effect=failed_pipeline))
            delivery = stack.enter_context(patch.object(main, '_send_email'))
            with self.assertRaisesRegex(RuntimeError, '1 failure'):
                main.run()
            delivery.assert_called_once()
            self.assertEqual(delivery.call_args.args[-1][0]['sub_id'], '123')
            main.Scheduler.return_value.shutdown.assert_called_once()

    def test_run_exits_failed_on_smtp_failure(self):
        with ExitStack() as stack:
            for name in ('Database', 'Transcriber', 'Summarizer', 'Emailer', 'Reporter',
                         'Scheduler', 'login_with_retry', 'ICourseClient', '_crawl_semester_catalog', '_drive_lectures'):
                stack.enter_context(patch.object(main, name))
            stack.enter_context(patch.object(config, 'COURSE_IDS', ['1']))
            stack.enter_context(patch.object(main, '_enumerate_lectures', return_value=[]))
            stack.enter_context(patch.object(main, '_send_email', side_effect=RuntimeError('SMTP failure')))
            with self.assertRaisesRegex(RuntimeError, '1 failure'):
                main.run()


if __name__ == '__main__':
    unittest.main()
