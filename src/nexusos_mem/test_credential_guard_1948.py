# -*- coding: utf-8 -*-
"""Task1948 凭据模式拦截单元测试（纯函数级，无 DB/网络依赖）。

覆盖 api_server.CREDENTIAL_PATTERNS 的 9 类特征命中 + 良性内容放行 +
title/content 双字段扫描。运行: python3 test_credential_guard_1948.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nexusos_mem.api_server import CREDENTIAL_PATTERNS, _credential_kinds  # noqa: E402


def test_pattern_inventory():
    kinds = {k for k, _ in CREDENTIAL_PATTERNS}
    assert kinds == {
        "sk_key", "password_eq", "token_eq", "aws_akia", "github_ghp",
        "google_aiza", "slack_xoxb", "jwt_eyJ", "private_key",
    }, kinds


def test_all_nine_patterns_fire():
    samples = {
        "sk_key": "key sk-abcdef1234567890abcd end",
        "password_eq": "PASSWORD=SuperSecret123456",
        "token_eq": "TOKEN=abcdef1234567890",
        "aws_akia": "AKIAABCDEFGHIJKLMNOP",  # gitleaks:allow — synthetic test fixture for the credential-rejection guard
        "github_ghp": "ghp_abcdef1234567890abcd",
        "google_aiza": "AIzaSyA1234567890abcdefghijklmnopqrstuv",  # gitleaks:allow — synthetic test fixture
        "slack_xoxb": "xoxb-1234567890-abcdef",
        "jwt_eyJ": "token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOjF9 tail",
        "private_key": "-----BEGIN RSA PRIVATE KEY-----",
    }
    for kind, sample in samples.items():
        hits = _credential_kinds(sample)
        assert kind in hits, f"{kind} 未命中: {sample!r} -> {hits}"


def test_benign_not_blocked():
    # 模板占位串（值 <12 位）不拦；字段名单独出现不拦；路径/表格线不拦
    assert _credential_kinds("模板 PASSWORD=change_me 占位") == []
    assert _credential_kinds("TOKEN= 空值不拦") == []
    assert "password_eq" not in _credential_kinds("字段名 ALIYUN_ROOT_PASSWORD 单独出现")
    assert _credential_kinds("路径 /opt/app/backend 与表格分隔线 -----") == []


def test_title_and_content_both_scanned():
    assert "sk_key" in _credential_kinds("干净正文", "标题含 sk-abcdef1234567890abcd")
    assert _credential_kinds("", "") == []
    assert _credential_kinds(None, None) == []


if __name__ == "__main__":
    test_pattern_inventory()
    test_all_nine_patterns_fire()
    test_benign_not_blocked()
    test_title_and_content_both_scanned()
    print("ALL_PASS")
