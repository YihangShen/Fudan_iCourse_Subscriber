"""Offline regressions for UIS TOTP second-factor login.

Run: python -m unittest discover -s tests -v
"""
import unittest
from unittest.mock import Mock, patch

import main
from src.api import webvpn
from src.api.webvpn import SecondFactorError, WebVPNSession, totp_code
from src.runtime import config

# RFC 6238 Appendix B SHA-1 seed "12345678901234567890" in Base32.
RFC_SECRET = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"


def _json_response(payload, status=200):
    resp = Mock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


class TotpCodeTests(unittest.TestCase):
    def test_rfc6238_vectors(self):
        # Last six digits of the RFC's eight-digit SHA-1 results.
        for ts, expected in [(59, "287082"), (1111111109, "081804"),
                             (1234567890, "005924"), (2000000000, "279037")]:
            self.assertEqual(totp_code(RFC_SECRET, ts), expected)

    def test_secret_normalisation(self):
        messy = " gezd gnbv gy3t qojq gezd gnbv gy3t qojq== "
        self.assertEqual(totp_code(messy, 59), "287082")

    def test_missing_or_invalid_secret(self):
        with self.assertRaises(SecondFactorError):
            totp_code("", 59)
        with self.assertRaises(SecondFactorError):
            totp_code("not-base32!", 59)


class FreshTotpWindowTests(unittest.TestCase):
    def setUp(self):
        webvpn._last_totp_counter = None

    def tearDown(self):
        webvpn._last_totp_counter = None

    def test_same_window_waits_for_next(self):
        clock = [61.0]
        sleeps = []

        def fake_sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds

        with patch.object(webvpn.time, "time", lambda: clock[0]), \
                patch.object(webvpn.time, "sleep", fake_sleep):
            first = webvpn.fresh_totp_code(RFC_SECRET)
            clock[0] = 75.0  # still window 2
            second = webvpn.fresh_totp_code(RFC_SECRET)
        self.assertEqual(first, totp_code(RFC_SECRET, 61))
        self.assertEqual(len(sleeps), 1)
        self.assertGreaterEqual(clock[0], 90.0)
        self.assertEqual(second, totp_code(RFC_SECRET, clock[0]))
        self.assertNotEqual(first, second)

    def test_new_window_does_not_wait(self):
        clock = [61.0]
        with patch.object(webvpn.time, "time", lambda: clock[0]), \
                patch.object(webvpn.time, "sleep") as sleep:
            webvpn.fresh_totp_code(RFC_SECRET)
            clock[0] = 95.0
            webvpn.fresh_totp_code(RFC_SECRET)
        sleep.assert_not_called()


class SecondFactorFlowTests(unittest.TestCase):
    """The 2026-11-09 failure: authExecute returned code 200 + second=true
    and no loginToken, so every login attempt failed."""

    def setUp(self):
        webvpn._last_totp_counter = None

    def tearDown(self):
        webvpn._last_totp_counter = None

    def _session(self, responses):
        vpn = WebVPNSession()
        vpn.session = Mock()
        vpn.session.post.side_effect = responses
        return vpn

    def test_password_then_totp_returns_login_token(self):
        challenge = {"code": "200", "second": True,
                     "moduleCodes": ["userAndOtp"], "requestNumber": "rn1",
                     "authChainCode": "chain2", "userName": "20000000000"}
        vpn = self._session([
            _json_response(challenge),
            _json_response({"code": "200", "loginToken": "tok"}),
        ])
        with patch.object(config, "TOTP_SECRET", RFC_SECRET):
            token = vpn._auth_execute("20000000000", "enc", "lck", "eid",
                                      "chain1", "chain_type")
        self.assertEqual(token, "tok")
        otp_payload = vpn.session.post.call_args_list[1].kwargs["json"]
        self.assertEqual(otp_payload["authModuleCode"], "userAndOtp")
        self.assertEqual(otp_payload["authChainCode"], "chain2")
        self.assertEqual(otp_payload["requestNumber"], "rn1")
        self.assertRegex(otp_payload["authPara"]["otpCode"], r"^\d{6}$")

    def test_missing_secret_stops_with_clear_error(self):
        challenge = {"code": "200", "second": True,
                     "moduleCodes": ["userAndOtp"], "requestNumber": "rn1",
                     "authChainCode": "chain2"}
        vpn = self._session([_json_response(challenge)])
        with patch.object(config, "TOTP_SECRET", ""), \
                self.assertRaisesRegex(SecondFactorError, "UIS_TOTP_SECRET"):
            vpn._auth_execute("s", "enc", "lck", "eid", "c", "chain_type")

    def test_rejected_totp_raises_second_factor_error(self):
        challenge = {"code": "200", "second": True,
                     "moduleCodes": ["userAndOtp"], "requestNumber": "rn1",
                     "authChainCode": "chain2"}
        vpn = self._session([
            _json_response(challenge),
            _json_response({"code": "500", "message": "bad otp"}),
        ])
        with patch.object(config, "TOTP_SECRET", RFC_SECRET), \
                self.assertRaises(SecondFactorError):
            vpn._auth_execute("s", "enc", "lck", "eid", "c", "chain_type")

    def test_login_with_retry_does_not_retry_second_factor_errors(self):
        fake = Mock()
        fake.login.side_effect = SecondFactorError("need secret")
        with patch.object(main, "WebVPNSession", return_value=fake), \
                patch.object(main.time, "sleep") as sleep, \
                self.assertRaises(SecondFactorError):
            main.login_with_retry(max_attempts=5)
        self.assertEqual(fake.login.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
