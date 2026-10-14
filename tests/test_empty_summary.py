import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.ai.summarizer import Summarizer, IncompleteSummaryError
from src.api.emailer import Emailer
from src.data.database import Database
from scripts.merge_db import merge


def response(content, reason='stop'):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=reason)], usage=None)


class SummaryValidationTests(unittest.TestCase):
    def setUp(self):
        self.summarizer = Summarizer.__new__(Summarizer)
        self.client = Mock()
        self.client.base_url = 'https://api.deepseek.com'
        self.summarizer.providers = [{'name': 'DeepSeek', 'models': ['deepseek-flash']}]
        self.summarizer._clients = {'DeepSeek': self.client}

    def test_empty_final_answer_retries_same_model(self):
        self.client.chat.completions.create.side_effect = [response(''), response('### 正文\n完整笔记')]
        summary, model = self.summarizer.summarize('逻辑', '课堂转录')
        self.assertIn('完整笔记', summary)
        self.assertEqual(self.client.chat.completions.create.call_count, 2)
        request = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request['extra_body']['thinking']['type'], 'disabled')

    def test_invalid_answers_never_succeed(self):
        for content, reason in [(None, 'stop'), (' \n\t', 'stop'), ('截断内容', 'length'), ('内容', 'content_filter')]:
            with self.subTest(content=content, reason=reason):
                self.client.chat.completions.create.return_value = response(content, reason)
                with self.assertRaisesRegex(RuntimeError, 'All LLM models failed'):
                    self.summarizer.summarize('逻辑', '课堂转录')

    def test_failed_model_falls_back(self):
        self.summarizer.providers[0]['models'].append('fallback')
        self.client.chat.completions.create.side_effect = [response(''), response(''), response('完整笔记')]
        self.assertEqual(self.summarizer.summarize('逻辑', '转录'), ('完整笔记', 'DeepSeek/fallback'))

    def test_custom_provider_does_not_receive_deepseek_options(self):
        self.client.base_url = 'https://example.com/v1'
        self.client.chat.completions.create.return_value = response('完整笔记')
        self.summarizer._call_llm(self.client, 'deepseek-flash', '逻辑', '转录')
        self.assertNotIn('extra_body', self.client.chat.completions.create.call_args.kwargs)

    def test_mailer_rejects_empty_note_before_smtp(self):
        with patch('src.api.emailer.smtplib.SMTP_SSL') as smtp:
            with self.assertRaisesRegex(ValueError, 'empty summary'):
                Emailer()._build_message([{'course_title': '逻辑', 'summary': ''}])
            smtp.assert_not_called()


class DatabaseRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = [str(Path(self.tmp.name) / name) for name in ('local.db','remote.db')]
        for path in self.paths:
            db = Database(path)
            db.upsert_course('1', '逻辑', '老师')
            db.insert_lecture('10', '1', '第1节', '2026-10-12')
            db.update_transcript('10', '已保存的转录')
            db.conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def corrupt(self, path):
        with sqlite3.connect(path) as db:
            db.execute("UPDATE lectures SET summary=' \n\t', processed_at='old', emailed_at='old'")

    def test_save_rejects_empty_output(self):
        db = Database(self.paths[0])
        try:
            with self.assertRaisesRegex(ValueError, 'empty summary'):
                db.update_summary('10', '', 'test')
            self.assertIsNone(db.get_lecture('10')['summary'])
        finally:
            db.conn.close()

    def test_old_empty_summary_reopens_without_losing_transcript(self):
        self.corrupt(self.paths[0])
        db = Database(self.paths[0])
        try:
            row = db.get_lecture('10')
            self.assertIsNone(row['processed_at'])
            self.assertIsNone(row['emailed_at'])
            self.assertEqual(row['transcript'], '已保存的转录')
            self.assertEqual(db.get_processed_sub_ids('1'), set())
            self.assertEqual(len(db.get_unprocessed_lectures('1')), 1)
            self.assertEqual(db.get_unsent_lectures(), [])
        finally:
            db.conn.close()

    def test_merge_does_not_restore_invalid_email_flag(self):
        self.corrupt(self.paths[1])
        local = Database(self.paths[0])
        local.update_summary('10', '补生成的完整笔记', 'test')
        local.mark_processed('10')
        local.conn.close()
        merge(*self.paths)
        remote = Database(self.paths[1])
        try:
            self.assertEqual(remote.get_lecture('10')['summary'], '补生成的完整笔记')
            self.assertIsNone(remote.get_lecture('10')['emailed_at'])
            self.assertEqual(len(remote.get_unsent_lectures()), 1)
        finally:
            remote.conn.close()

    def test_invalid_local_cannot_overwrite_good_remote_summary(self):
        self.corrupt(self.paths[0])
        remote = Database(self.paths[1])
        remote.update_summary('10', '正确笔记', 'test')
        remote.mark_processed('10')
        remote.mark_emailed('10')
        remote.conn.close()
        merge(*self.paths)
        remote = Database(self.paths[1])
        try:
            self.assertEqual(remote.get_lecture('10')['summary'], '正确笔记')
            self.assertTrue(remote.get_lecture('10')['emailed_at'])
        finally:
            remote.conn.close()

if __name__ == '__main__':
    unittest.main()
